import argparse
import hashlib
import json
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLANTDOC_DIR = PROJECT_ROOT / "dataset" / "PlantDoc-Object-Detection-Dataset-master"
DETECTOR_DIR = PROJECT_ROOT / "models" / "plant_detector_v2"
CLASSIFIER_DATASET_DIR = PROJECT_ROOT / "dataset" / "classifier_v2"
CLASSIFIER_MODEL_DIR = PROJECT_ROOT / "models" / "quantum_model_v2"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSIFIER_CLASSES = ("disease", "healthy", "non_crop", "pest")


def _read_text(element, path, default=""):
    value = element.findtext(path, default=default)
    return value.strip() if value else default


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_image(xml_path, embedded_name, images_by_name, images_by_stem):
    filename = Path(embedded_name.replace("\\", "/")).name if embedded_name else ""
    candidates = images_by_name.get(filename.casefold(), [])
    if len(candidates) == 1:
        return candidates[0], "embedded_filename"
    if len(candidates) > 1:
        return None, "ambiguous_embedded_filename"

    xml_stem = xml_path.stem.casefold()
    candidates = images_by_stem.get(xml_stem, [])
    if len(candidates) == 1:
        return candidates[0], "xml_basename_fallback"
    if len(candidates) > 1:
        return None, "ambiguous_xml_basename"
    return None, "missing_image"


def _collect_split(split_dir, split_name):
    images = [
        path for path in split_dir.iterdir()
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    ]
    images_by_name = defaultdict(list)
    images_by_stem = defaultdict(list)
    for image_path in images:
        images_by_name[image_path.name.casefold()].append(image_path)
        images_by_stem[image_path.stem.casefold()].append(image_path)

    statistics = {
        "source_images": len(images),
        "xml_files": 0,
        "embedded_filename_matches": 0,
        "xml_basename_fallback_matches": 0,
        "missing_or_ambiguous_image_references": 0,
        "invalid_xml_files": 0,
        "image_size_mismatches": 0,
        "ignored_non_leaf_objects": 0,
        "invalid_leaf_boxes": 0,
    }
    samples_by_path = {}
    ignored_categories = Counter()

    for xml_path in sorted(split_dir.glob("*.xml")):
        statistics["xml_files"] += 1
        try:
            annotation = ET.parse(xml_path).getroot()
        except ET.ParseError:
            statistics["invalid_xml_files"] += 1
            continue

        image_path, match_type = _resolve_image(
            xml_path,
            _read_text(annotation, "filename"),
            images_by_name,
            images_by_stem,
        )
        if image_path is None:
            statistics["missing_or_ambiguous_image_references"] += 1
            continue
        statistics[
            "embedded_filename_matches"
            if match_type == "embedded_filename"
            else "xml_basename_fallback_matches"
        ] += 1

        try:
            with Image.open(image_path) as image:
                width, height = image.size
                image.verify()
        except (OSError, ValueError):
            statistics["invalid_xml_files"] += 1
            continue

        declared_size = annotation.find("size")
        if declared_size is not None:
            try:
                declared_width = int(_read_text(declared_size, "width"))
                declared_height = int(_read_text(declared_size, "height"))
            except ValueError:
                declared_width = declared_height = 0
            if (declared_width, declared_height) != (width, height):
                statistics["image_size_mismatches"] += 1

        sample = samples_by_path.setdefault(image_path, {
            "source_split": split_name,
            "image_path": image_path,
            "width": width,
            "height": height,
            "boxes": [],
        })
        for obj in annotation.findall("object"):
            category = _read_text(obj, "name")
            if "leaf" not in category.casefold():
                statistics["ignored_non_leaf_objects"] += 1
                ignored_categories[category or "(missing label)"] += 1
                continue

            box = obj.find("bndbox")
            try:
                xmin = float(_read_text(box, "xmin"))
                ymin = float(_read_text(box, "ymin"))
                xmax = float(_read_text(box, "xmax"))
                ymax = float(_read_text(box, "ymax"))
            except (AttributeError, TypeError, ValueError):
                statistics["invalid_leaf_boxes"] += 1
                continue

            if (
                xmin < 0
                or ymin < 0
                or xmax > width
                or ymax > height
                or xmax <= xmin
                or ymax <= ymin
            ):
                statistics["invalid_leaf_boxes"] += 1
                continue

            # Pascal VOC boxes are one-based; YOLO boxes use zero-based image coordinates.
            x1 = max(0.0, xmin - 1.0)
            y1 = max(0.0, ymin - 1.0)
            x2 = xmax
            y2 = ymax
            if x2 <= x1 or y2 <= y1:
                statistics["invalid_leaf_boxes"] += 1
                continue
            sample["boxes"].append((x1, y1, x2, y2))

    samples = [sample for sample in samples_by_path.values() if sample["boxes"]]
    return samples, statistics, dict(ignored_categories)


def _write_yolo_label(label_path, sample):
    width = sample["width"]
    height = sample["height"]
    rows = []
    for x1, y1, x2, y2 in sample["boxes"]:
        center_x = ((x1 + x2) / 2) / width
        center_y = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        rows.append(
            f"0 {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )
    label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _partition_train_samples(samples, test_hashes, validation_fraction):
    training = []
    validation = []
    excluded_duplicate_count = 0
    for sample in samples:
        digest = sample["sha256"]
        if digest in test_hashes:
            excluded_duplicate_count += 1
            continue
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        if bucket < validation_fraction:
            validation.append(sample)
        else:
            training.append(sample)
    return training, validation, excluded_duplicate_count


def prepare_plantdoc(
    source_dir=PLANTDOC_DIR,
    detector_dir=DETECTOR_DIR,
    classifier_dataset_dir=CLASSIFIER_DATASET_DIR,
    classifier_model_dir=CLASSIFIER_MODEL_DIR,
    validation_fraction=0.15,
):
    source_dir = Path(source_dir)
    detector_dir = Path(detector_dir)
    detector_dataset_dir = detector_dir / "dataset"
    classifier_dataset_dir = Path(classifier_dataset_dir)
    classifier_model_dir = Path(classifier_model_dir)

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be greater than 0 and less than 1")
    if detector_dataset_dir.exists():
        raise FileExistsError(
            f"Prepared detector dataset already exists; refusing to overwrite it: "
            f"{detector_dataset_dir}"
        )
    for split_name in ("TRAIN", "TEST"):
        if not (source_dir / split_name).is_dir():
            raise FileNotFoundError(f"PlantDoc split directory is missing: {source_dir / split_name}")

    train_samples, train_stats, ignored_train = _collect_split(
        source_dir / "TRAIN", "TRAIN"
    )
    test_samples, test_stats, ignored_test = _collect_split(
        source_dir / "TEST", "TEST"
    )
    for sample in train_samples + test_samples:
        sample["sha256"] = _file_digest(sample["image_path"])

    test_hashes = {sample["sha256"] for sample in test_samples}
    training, validation, duplicate_train_count = _partition_train_samples(
        train_samples, test_hashes, validation_fraction
    )
    split_samples = {
        "train": training,
        "validation": validation,
        "test": test_samples,
    }

    report = {
        "source": str(source_dir),
        "detector_class_order": ["leaf"],
        "classifier_class_order": ["Healthy", "Pest", "Disease", "Non-Crop"],
        "detector_labeling_rule": (
            "Every source object explicitly labeled as a leaf is a generic leaf box. "
            "Source disease/species names are not used as pest or classifier labels."
        ),
        "validation_fraction_of_usable_training_images": validation_fraction,
        "excluded_exact_train_test_duplicates": duplicate_train_count,
        "source_splits": {
            "TRAIN": {
                **train_stats,
                "usable_annotated_images": len(train_samples),
                "leaf_boxes": sum(len(sample["boxes"]) for sample in train_samples),
                "ignored_non_leaf_categories": ignored_train,
            },
            "TEST": {
                **test_stats,
                "usable_annotated_images": len(test_samples),
                "leaf_boxes": sum(len(sample["boxes"]) for sample in test_samples),
                "ignored_non_leaf_categories": ignored_test,
            },
        },
        "prepared_splits": {
            name: {
                "images": len(samples),
                "leaf_boxes": sum(len(sample["boxes"]) for sample in samples),
            }
            for name, samples in split_samples.items()
        },
        "classifier_folder_counts_before_user_labeling": {},
    }

    detector_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dataset-stage-", dir=detector_dir) as staging:
        staging_dir = Path(staging)
        for split_name, samples in split_samples.items():
            image_dir = staging_dir / "images" / split_name
            label_dir = staging_dir / "labels" / split_name
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            for sample in samples:
                source_image = sample["image_path"]
                output_name = (
                    f"{source_image.stem[:80]}_{sample['sha256'][:12]}"
                    f"{source_image.suffix.lower()}"
                )
                shutil.copy2(source_image, image_dir / output_name)
                _write_yolo_label(label_dir / f"{Path(output_name).stem}.txt", sample)

        yaml_text = (
            f'path: "{detector_dataset_dir.resolve().as_posix()}"\n'
            "train: images/train\n"
            "val: images/validation\n"
            "test: images/test\n"
            "names:\n"
            "  0: leaf\n"
        )
        (staging_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
        report["classifier_folder_counts_before_user_labeling"] = _create_classifier_folders(
            classifier_dataset_dir
        )
        (staging_dir / "preparation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        os.replace(staging_dir, detector_dataset_dir)

    classifier_model_dir.mkdir(parents=True, exist_ok=True)
    return report


def _create_classifier_folders(classifier_dataset_dir):
    counts = {}
    for split_name in ("train", "validation", "test"):
        counts[split_name] = {}
        for class_name in CLASSIFIER_CLASSES:
            class_dir = classifier_dataset_dir / split_name / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            counts[split_name][class_name] = sum(
                path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
                for path in class_dir.iterdir()
            )
    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Prepare PlantDoc leaf boxes for a new YOLO detector, without training."
    )
    parser.add_argument("--source-dir", type=Path, default=PLANTDOC_DIR)
    parser.add_argument("--detector-dir", type=Path, default=DETECTOR_DIR)
    parser.add_argument("--classifier-dataset-dir", type=Path, default=CLASSIFIER_DATASET_DIR)
    parser.add_argument("--classifier-model-dir", type=Path, default=CLASSIFIER_MODEL_DIR)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    report = prepare_plantdoc(
        source_dir=args.source_dir,
        detector_dir=args.detector_dir,
        classifier_dataset_dir=args.classifier_dataset_dir,
        classifier_model_dir=args.classifier_model_dir,
        validation_fraction=args.validation_fraction,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

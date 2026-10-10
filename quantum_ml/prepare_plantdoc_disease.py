import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "dataset" / "PlantDoc-Object-Detection-Dataset-master"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "plantdoc_disease_v2"
SPLITS = {
    "train": "TRAIN",
    "test": "TEST",
}
DISEASE_LABELS = {
    "Apple rust leaf",
    "Apple Scab Leaf",
    "Bell_pepper leaf spot",
    "Corn Gray leaf spot",
    "Corn leaf blight",
    "Corn rust leaf",
    "grape leaf black rot",
    "Potato leaf early blight",
    "Potato leaf late blight",
    "Squash Powdery mildew leaf",
    "Tomato Early blight leaf",
    "Tomato leaf bacterial spot",
    "Tomato leaf late blight",
    "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus",
    "Tomato mold leaf",
    "Tomato Septoria leaf spot",
}
UNSPECIFIED_LEAF_LABELS = {
    "Apple leaf",
    "Bell_pepper leaf",
    "Blueberry leaf",
    "Cherry leaf",
    "grape leaf",
    "Peach leaf",
    "Potato leaf",
    "Raspberry leaf",
    "Soyabean leaf",
    "Strawberry leaf",
    "Tomato leaf",
}
PEST_LABELS = {"Tomato two spotted spider mites leaf"}
CLASS_NAMES = ("Disease", "Unspecified Leaf")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _image_index(directory):
    images = defaultdict(list)
    for path in directory.iterdir():
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS:
            images[path.name.casefold()].append(path)
    return images


def _read_annotations(source_dir, split_name, report):
    directory = source_dir / SPLITS[split_name]
    csv_path = source_dir / f"{split_name}_labels.csv"
    image_index = _image_index(directory)
    annotations = defaultdict(list)
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            category = row.get("class", "").strip()
            filename = Path(row.get("filename", "").replace("\\", "/")).name
            candidates = image_index.get(filename.casefold(), [])
            if not candidates:
                report["missing_images"] += 1
                continue
            if len(candidates) != 1:
                report["ambiguous_images"] += 1
                continue
            if category in PEST_LABELS:
                report["excluded_pest_annotations"] += 1
                continue
            if category in DISEASE_LABELS:
                target_class = "disease"
            elif category in UNSPECIFIED_LEAF_LABELS:
                target_class = "unspecified_leaf"
            else:
                report["unmapped_categories"][category or "(empty)"] += 1
                continue
            try:
                box = tuple(float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
            except (KeyError, TypeError, ValueError) as error:
                report["invalid_boxes"] += 1
                continue
            annotations[candidates[0]].append((target_class, box))
    return annotations


def _split_train_images(image_paths, validation_fraction, test_hashes, report):
    training = []
    validation = []
    for path in sorted(image_paths):
        digest = _digest(path)
        if digest in test_hashes:
            report["duplicate_train_images_excluded"] += 1
            continue
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        (validation if bucket < validation_fraction else training).append(path)
    return training, validation


def _save_crops(source_images, annotations, output_dir, split_name, report):
    split_counts = Counter()
    for image_path in source_images:
        try:
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
        except (OSError, ValueError, UnidentifiedImageError):
            report["unreadable_images"] += 1
            continue
        width, height = image.size
        image_digest = _digest(image_path)
        for index, (class_name, coordinates) in enumerate(annotations[image_path]):
            xmin, ymin, xmax, ymax = coordinates
            left = max(0, int(xmin) - 1)
            top = max(0, int(ymin) - 1)
            right = min(width, int(xmax))
            bottom = min(height, int(ymax))
            if right <= left or bottom <= top:
                report["invalid_boxes"] += 1
                continue
            destination_dir = output_dir / split_name / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            crop = image.crop((left, top, right, bottom))
            crop_name = hashlib.sha256(
                f"{image_digest}:{xmin}:{ymin}:{xmax}:{ymax}:{index}".encode("utf-8")
            ).hexdigest()[:20] + ".jpg"
            crop.save(destination_dir / crop_name, format="JPEG", quality=95)
            split_counts[class_name] += 1
            report["source_images_used"][split_name].add(image_path.name)
    return {name: split_counts[name] for name in ("disease", "unspecified_leaf")}


def prepare_plantdoc_disease(
    source_dir=SOURCE_DIR,
    output_dir=OUTPUT_DIR,
    validation_fraction=0.15,
):
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be greater than 0 and less than 1")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prepared disease data: {output_dir}")
    for split_name in SPLITS:
        csv_path = source_dir / f"{split_name}_labels.csv"
        image_dir = source_dir / SPLITS[split_name]
        if not csv_path.is_file() or not image_dir.is_dir():
            raise FileNotFoundError(f"PlantDoc {split_name} annotations or images are missing")

    report = {
        "source_dataset": "PlantDoc",
        "class_names": list(CLASS_NAMES),
        "class_meanings": {
            "Disease": "PlantDoc labels that explicitly name a disease",
            "Unspecified Leaf": "PlantDoc labels naming a leaf without a disease",
        },
        "pest_labels_excluded": sorted(PEST_LABELS),
        "train_validation_fraction": validation_fraction,
        "missing_images": 0,
        "ambiguous_images": 0,
        "unreadable_images": 0,
        "invalid_boxes": 0,
        "excluded_pest_annotations": 0,
        "duplicate_train_images_excluded": 0,
        "unmapped_categories": Counter(),
        "source_images_used": {name: set() for name in ("train", "validation", "test")},
    }
    train_annotations = _read_annotations(source_dir, "train", report)
    test_annotations = _read_annotations(source_dir, "test", report)
    test_hashes = {_digest(path) for path, boxes in test_annotations.items() if boxes}
    training_images, validation_images = _split_train_images(
        train_annotations, validation_fraction, test_hashes, report
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": _save_crops(
            training_images, train_annotations, output_dir, "train", report
        ),
        "validation": _save_crops(
            validation_images, train_annotations, output_dir, "validation", report
        ),
        "test": _save_crops(
            sorted(test_annotations), test_annotations, output_dir, "test", report
        ),
    }
    if any(counts["train"][name] == 0 for name in ("disease", "unspecified_leaf")):
        raise RuntimeError("PlantDoc preparation produced an empty training class")
    if any(counts["test"][name] == 0 for name in ("disease", "unspecified_leaf")):
        raise RuntimeError("PlantDoc held-out TEST split must contain both supported labels")

    report["image_counts"] = counts
    report["source_image_counts"] = {
        name: len(paths) for name, paths in report["source_images_used"].items()
    }
    report["source_images_used"] = {
        name: sorted(paths) for name, paths in report["source_images_used"].items()
    }
    report["unmapped_categories"] = dict(report["unmapped_categories"])
    (output_dir / "preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Prepare separate disease/unspecified-leaf crops from PlantDoc CSV boxes."
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    report = prepare_plantdoc_disease(
        args.source_dir, args.output_dir, args.validation_fraction
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

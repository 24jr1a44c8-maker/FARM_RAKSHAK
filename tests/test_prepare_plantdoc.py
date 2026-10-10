import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image

from quantum_ml.prepare_plantdoc import prepare_plantdoc
from quantum_ml.train_quantum_v2 import CLASS_NAMES, load_split, train_pipeline
from quantum_ml.train_leaf_detector_v2 import YoloLeafDataset, _measure_predictions


def write_annotation(path, image_name, image_size, objects):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = image_name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(image_size[0])
    ET.SubElement(size, "height").text = str(image_size[1])
    for name, box in objects:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = name
        bounds = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), box):
            ET.SubElement(bounds, key).text = str(value)
    ET.ElementTree(annotation).write(path, encoding="utf-8")


class PlantDocPreparationTests(unittest.TestCase):
    def test_leaf_detector_loader_reads_normalized_yolo_boxes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_dir = root / "images" / "train"
            label_dir = root / "labels" / "train"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            Image.new("RGB", (100, 80), "green").save(image_dir / "leaf.jpg")
            (label_dir / "leaf.txt").write_text("0 0.5 0.5 0.4 0.5\n", encoding="utf-8")

            image, target = YoloLeafDataset(root, "train")[0]

        self.assertEqual(tuple(image.shape), (3, 80, 100))
        self.assertEqual(target["boxes"].tolist(), [[30.0, 20.0, 70.0, 60.0]])
        self.assertEqual(target["labels"].tolist(), [1])

    def test_leaf_detector_evaluation_counts_duplicate_boxes_as_false_positives(self):
        predictions = [{
            "boxes": torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32),
            "labels": torch.tensor([1, 1]),
            "scores": torch.tensor([0.9, 0.8]),
            "ground_truth": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
        }]

        metrics = _measure_predictions(predictions, score_threshold=0.5)

        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 0)

    def test_classifier_loader_uses_new_dataset_path_and_fixed_class_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "classifier_v2"
            for class_name in CLASS_NAMES:
                class_dir = dataset_dir / "train" / class_name.lower().replace("-", "_")
                class_dir.mkdir(parents=True)
                Image.new("RGB", (16, 16), "green").save(class_dir / "sample.jpg")

            images, labels, counts = load_split("train", dataset_dir)

        self.assertEqual(len(images), 4)
        self.assertEqual(labels.tolist(), [0, 1, 2, 3])
        self.assertEqual(counts, {name: 1 for name in CLASS_NAMES})

    def test_v2_trainer_refuses_to_overwrite_existing_model_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "classifier_v2"
            for class_name in CLASS_NAMES:
                class_dir = dataset_dir / "train" / class_name.lower().replace("-", "_")
                class_dir.mkdir(parents=True)
                Image.new("RGB", (16, 16), "green").save(class_dir / "sample.jpg")

            model_dir = root / "quantum_model_v2"
            model_dir.mkdir()
            marker = model_dir / "keep.txt"
            marker.write_text("do not overwrite", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                train_pipeline(dataset_dir=dataset_dir, model_dir=model_dir)

            self.assertEqual(marker.read_text(encoding="utf-8"), "do not overwrite")

    def test_xml_is_prepared_as_leaf_boxes_not_classifier_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "PlantDoc"
            train = source / "TRAIN"
            test = source / "TEST"
            train.mkdir(parents=True)
            test.mkdir()

            train_image = train / "mite-damage_saved.jpg"
            Image.new("RGB", (100, 80), "green").save(train_image)
            write_annotation(
                train / "mite-damage_saved.xml",
                "mite-damage?download=1.jpg",
                (100, 80),
                [
                    ("Tomato early blight leaf", (1, 2, 41, 42)),
                    ("Tomato two spotted spider mites leaf", (50, 20, 90, 70)),
                    ("person", (2, 2, 20, 20)),
                ],
            )

            test_image = test / "heldout.jpg"
            Image.new("RGB", (100, 80), "blue").save(test_image)
            write_annotation(
                test / "heldout.xml",
                "heldout.jpg",
                (100, 80),
                [("Apple leaf", (10, 10, 80, 60))],
            )

            detector_dir = root / "models" / "plant_detector_v2"
            classifier_data = root / "dataset" / "classifier_v2"
            classifier_models = root / "models" / "quantum_model_v2"
            report = prepare_plantdoc(
                source_dir=source,
                detector_dir=detector_dir,
                classifier_dataset_dir=classifier_data,
                classifier_model_dir=classifier_models,
                validation_fraction=0.5,
            )

            self.assertEqual(report["detector_class_order"], ["leaf"])
            self.assertEqual(report["source_splits"]["TRAIN"]["leaf_boxes"], 2)
            self.assertEqual(
                report["source_splits"]["TRAIN"]["ignored_non_leaf_categories"], {"person": 1}
            )
            self.assertEqual(report["prepared_splits"]["test"]["leaf_boxes"], 1)
            self.assertEqual(
                report["classifier_folder_counts_before_user_labeling"]["train"],
                {"disease": 0, "healthy": 0, "non_crop": 0, "pest": 0},
            )
            self.assertTrue(classifier_models.is_dir())
            self.assertFalse((detector_dir / "best.pt").exists())
            yaml_text = (detector_dir / "dataset" / "data.yaml").read_text(encoding="utf-8")
            self.assertIn(f'path: "{(detector_dir / "dataset").resolve().as_posix()}"', yaml_text)
            self.assertIn("  0: leaf", yaml_text)

            label_files = list((detector_dir / "dataset" / "labels").glob("*/*.txt"))
            all_label_rows = [
                row
                for label_file in label_files
                for row in label_file.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(all_label_rows), 3)
            self.assertTrue(all(row.startswith("0 ") for row in all_label_rows))

    def test_refuses_to_overwrite_prepared_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "PlantDoc"
            (source / "TRAIN").mkdir(parents=True)
            (source / "TEST").mkdir()
            detector_dir = root / "models" / "plant_detector_v2"
            prepared_dataset = detector_dir / "dataset"
            prepared_dataset.mkdir(parents=True)

            with self.assertRaises(FileExistsError):
                prepare_plantdoc(
                    source_dir=source,
                    detector_dir=detector_dir,
                    classifier_dataset_dir=root / "dataset" / "classifier_v2",
                    classifier_model_dir=root / "models" / "quantum_model_v2",
                )


if __name__ == "__main__":
    unittest.main()

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from quantum_ml.predict_plantdoc_disease import (
    disease_model_readiness,
    predict_plantdoc_disease,
)
from quantum_ml.prepare_plantdoc_disease import prepare_plantdoc_disease
from quantum_ml.train_plantdoc_disease import CLASS_NAMES, load_split, train_pipeline


class PlantDocDiseaseTests(unittest.TestCase):
    def _make_source(self, root, split_name, count_per_class):
        image_dir = root / split_name.upper()
        image_dir.mkdir(parents=True)
        csv_path = root / f"{split_name}_labels.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=("filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"),
            )
            writer.writeheader()
            classes = (
                ("Corn leaf blight", "disease", "red"),
                ("Tomato leaf", "unspecified_leaf", "green"),
            )
            for category, _, color in classes:
                for index in range(count_per_class):
                    filename = f"{split_name}-{category.replace(' ', '-')}-{index}.jpg"
                    offset = 0 if split_name == "train" else 70
                    pixel = (
                        (150 + index + offset, 0, 0)
                        if color == "red"
                        else (0, 100 + index + offset, 0)
                    )
                    Image.new("RGB", (32, 32), pixel).save(image_dir / filename)
                    writer.writerow({
                        "filename": filename,
                        "width": 32,
                        "height": 32,
                        "class": category,
                        "xmin": 1,
                        "ymin": 1,
                        "xmax": 32,
                        "ymax": 32,
                    })
            if split_name == "train":
                pest_name = "train-mite.jpg"
                Image.new("RGB", (32, 32), "yellow").save(image_dir / pest_name)
                writer.writerow({
                    "filename": pest_name,
                    "width": 32,
                    "height": 32,
                    "class": "Tomato two spotted spider mites leaf",
                    "xmin": 1,
                    "ymin": 1,
                    "xmax": 32,
                    "ymax": 32,
                })

    def test_preparation_keeps_disease_unknown_and_pest_labels_distinct(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "PlantDoc"
            source.mkdir()
            self._make_source(source, "train", 60)
            self._make_source(source, "test", 4)
            output = root / "prepared"

            report = prepare_plantdoc_disease(
                source_dir=source,
                output_dir=output,
                validation_fraction=0.2,
            )

            self.assertEqual(report["class_names"], list(CLASS_NAMES))
            self.assertEqual(report["excluded_pest_annotations"], 1)
            self.assertFalse(report["unmapped_categories"])
            self.assertGreater(report["image_counts"]["train"]["disease"], 0)
            self.assertGreater(report["image_counts"]["validation"]["disease"], 0)
            self.assertEqual(report["image_counts"]["test"]["disease"], 4)
            self.assertFalse((output / "train" / "pest").exists())
            self.assertTrue((output / "preparation_report.json").is_file())

    def test_preparation_refuses_to_overwrite_existing_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "PlantDoc"
            source.mkdir()
            self._make_source(source, "train", 1)
            self._make_source(source, "test", 1)
            output = root / "prepared"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_plantdoc_disease(source_dir=source, output_dir=output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_loader_uses_disease_then_unspecified_leaf_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for folder, color in (("disease", "red"), ("unspecified_leaf", "green")):
                class_dir = root / "train" / folder
                class_dir.mkdir(parents=True)
                Image.new("RGB", (16, 16), color).save(class_dir / "sample.jpg")

            images, labels, counts = load_split("train", root)

        self.assertEqual(len(images), 2)
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(counts, {"Disease": 1, "Unspecified Leaf": 1})

    def test_disease_model_readiness_requires_versioned_hashed_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            artifacts = {}
            for filename in ("scaler.pkl", "pca.pkl", "vqc.model"):
                artifact = model_dir / filename
                artifact.write_bytes(filename.encode("ascii"))
                artifacts[filename] = hashlib.sha256(artifact.read_bytes()).hexdigest()
            metadata = {
                "model_metadata_version": 1,
                "class_names": list(CLASS_NAMES),
                "training_samples": 4,
                "training_counts": {name: 2 for name in CLASS_NAMES},
                "artifacts": artifacts,
                "inference_approved": False,
            }
            (model_dir / "model_metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with (
                patch("quantum_ml.predict_plantdoc_disease.MODEL_DIR", model_dir),
                patch(
                    "quantum_ml.predict_plantdoc_disease.METADATA_PATH",
                    model_dir / "model_metadata.json",
                ),
            ):
                ready, message = disease_model_readiness()
                self.assertFalse(ready)
                self.assertIn("not approved", message)
                metadata["inference_approved"] = True
                (model_dir / "model_metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                ready, message = disease_model_readiness()

        self.assertTrue(ready, message)

    def test_unspecified_leaf_prediction_never_becomes_healthy(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        cnn = Mock()
        cnn.extract_features.return_value = [0.0, 0.0, 0.0, 0.0]
        reducer = Mock()
        reducer.reduce.return_value = [[0.0, 0.0, 0.0, 0.0]]
        vqc = SimpleNamespace(vqc=Mock())
        vqc.vqc.predict_proba.return_value = [[0.1, 0.9]]
        with (
            patch(
                "quantum_ml.predict_plantdoc_disease.disease_model_readiness",
                return_value=(True, None),
            ),
            patch(
                "quantum_ml.predict_plantdoc_disease._get_models",
                return_value=(cnn, reducer, vqc),
            ),
        ):
            result = predict_plantdoc_disease(Image.new("RGB", (16, 16)))

        self.assertEqual(result["prediction"], "UNCERTAIN")
        self.assertEqual(result["matched_class"], "Unspecified Leaf")
        self.assertEqual(result["classification_scope"], "disease_only")

    def test_disease_trainer_refuses_to_overwrite_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "existing"
            model_dir.mkdir()
            marker = model_dir / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                train_pipeline(model_dir=model_dir)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()

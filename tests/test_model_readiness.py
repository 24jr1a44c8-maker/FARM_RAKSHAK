import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from quantum_ml.predict_quantum import model_readiness


class ModelReadinessTests(unittest.TestCase):
    def test_legacy_three_class_metadata_is_not_trusted(self):
        legacy_metadata = {
            "class_names": ["Healthy", "Pest", "Disease"],
            "training_samples": 120,
        }
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "model_metadata.json"
            metadata_path.write_text(json.dumps(legacy_metadata), encoding="utf-8")
            with patch("quantum_ml.predict_quantum.METADATA_PATH", str(metadata_path)):
                ready, message = model_readiness()

        self.assertFalse(ready)
        self.assertIn("required training schema", message)

    def test_readiness_requires_matching_four_class_manifest_and_artifacts(self):
        class_names = ["Healthy", "Pest", "Disease", "Non-Crop"]
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            artifact_hashes = {}
            for filename in ("scaler.pkl", "pca.pkl", "vqc.model"):
                artifact = model_dir / filename
                artifact.write_bytes(filename.encode("ascii"))
                artifact_hashes[filename] = hashlib.sha256(artifact.read_bytes()).hexdigest()

            metadata = {
                "model_metadata_version": 2,
                "class_names": class_names,
                "training_samples": 4,
                "training_counts": {name: 1 for name in class_names},
                "artifacts": artifact_hashes,
            }
            metadata_path = model_dir / "model_metadata.json"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with (
                patch("quantum_ml.predict_quantum.METADATA_PATH", metadata_path),
                patch("quantum_ml.predict_quantum.MODEL_DIR", model_dir),
            ):
                ready, message = model_readiness()
                self.assertTrue(ready, message)

                (model_dir / "vqc.model").write_bytes(b"changed")
                ready, message = model_readiness()

        self.assertFalse(ready)
        self.assertIn("trained artifacts", message)


if __name__ == "__main__":
    unittest.main()
import unittest
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from quantum_ml.plant_detector import PlantDetector, PlantDetectorNotReady


class PlantDetectorReadinessTests(unittest.TestCase):
    def test_torchvision_leaf_checkpoint_requires_explicit_leaf_class_order(self):
        with tempfile.NamedTemporaryFile(suffix=".pth") as model_file:
            model = Mock()
            with (
                patch.dict(os.environ, {"PLANT_DETECTOR_MODEL": model_file.name}),
                patch.object(PlantDetector, "_load_model", return_value=model) as loader,
            ):
                detector = PlantDetector()
                self.assertEqual(detector.readiness(), (True, None))
                self.assertEqual(detector.confidence_threshold, 0.35)
                loader.assert_called_once()

    def test_detector_refuses_frames_without_configured_weights(self):
        with patch.dict(os.environ, {"PLANT_DETECTOR_MODEL": "missing-detector.pt"}):
            detector = PlantDetector()
            image = Image.new("RGB", (64, 64), "green")

            ready, message = detector.readiness()
            self.assertFalse(ready)
            self.assertIn("weights are missing", message)
            with self.assertRaises(PlantDetectorNotReady):
                detector.detect_regions(image)

    def test_detector_load_error_is_reported_and_cached(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with (
                patch.dict(os.environ, {"PLANT_DETECTOR_MODEL": model_file.name}),
                patch.object(PlantDetector, "_load_model", side_effect=RuntimeError("bad weights")) as loader,
            ):
                detector = PlantDetector()
                first = detector.readiness()
                second = detector.readiness()

        self.assertEqual(first, (False, "Plant detector failed to load: bad weights"))
        self.assertEqual(second, first)
        loader.assert_called_once()

    def test_detector_returns_only_configured_plant_boxes_from_model_coordinates(self):
        class TensorValues:
            def __init__(self, values):
                self.values = values

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.values

        names = {0: "plant", 1: "person"}
        boxes = SimpleNamespace(
            xyxy=TensorValues([[3.5, 4, 40, 48], [5, 6, 20, 30]]),
            cls=TensorValues([0, 1]),
            conf=TensorValues([0.93, 0.99]),
        )
        model = Mock()
        model.names = names
        model.predict.return_value = [SimpleNamespace(boxes=boxes, names=names)]

        with tempfile.NamedTemporaryFile(suffix=".pt") as model_file:
            with (
                patch.dict(os.environ, {"PLANT_DETECTOR_MODEL": model_file.name}),
                patch.object(PlantDetector, "_load_model", return_value=model),
            ):
                detector = PlantDetector()
                image = Image.new("RGB", (64, 64), "green")
                results = detector.detect_regions(image)

        self.assertEqual(
            results,
            [{
                "box": [3.5, 4.0, 40.0, 48.0],
                "detector_score": 0.93,
                "detector_class": "plant",
            }],
        )
        model.predict.assert_called_once_with(source=image, conf=0.25, verbose=False)


if __name__ == "__main__":
    unittest.main()
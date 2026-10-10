import base64
import io
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from backend.main import FrameRequest, get_zone, get_zones, health, predict, process_frame, reset_zone, scan


def encoded_test_image():
    image_buffer = io.BytesIO()
    Image.new("RGB", (32, 32), "green").save(image_buffer, format="PNG")
    return base64.b64encode(image_buffer.getvalue()).decode("ascii")


class ApiContractTests(unittest.TestCase):
    def test_zone_endpoints_return_six_real_zone_records(self):
        self.assertEqual(len(get_zones()["zones"]), 6)
        self.assertEqual(get_zone(1)["zone"], "Zone 1")

    def test_scan_requires_a_zone(self):
        with self.assertRaises(Exception):
            process_frame(FrameRequest(image="eA=="), record_scan=True)

    def test_unready_classifier_returns_leaf_detection_without_recording_health(self):
        before = get_zone(1)["plants_scanned"]
        detector = Mock()
        detector.readiness.return_value = (True, None)
        detector.detect_regions.return_value = [
            {
                "box": [2, 3, 22, 28],
                "detector_score": 0.9,
                "detector_class": "leaf",
            }
        ]
        with (
            patch("backend.main.get_plant_detector", return_value=detector),
            patch("backend.main.model_readiness", return_value=(False, "training required")),
            patch(
                "backend.main.disease_model_readiness",
                return_value=(False, "PlantDoc disease model is research-only and not approved for live inference"),
            ),
        ):
            result = scan(FrameRequest(image=encoded_test_image(), zone_id=1))
        self.assertEqual(result["status"], "CLASSIFICATION_UNAVAILABLE")
        self.assertEqual(
            result["message"],
            "PlantDoc disease model is research-only and not approved for live inference",
        )
        self.assertTrue(result["plant_detected"])
        self.assertEqual(result["detections"][0]["prediction"], "Leaf")
        self.assertIsNone(result["detections"][0]["confidence"])
        self.assertEqual(get_zone(1)["plants_scanned"], before)

    def test_disease_only_classifier_returns_cautious_match_without_zone_health_update(self):
        detector = Mock()
        detector.readiness.return_value = (True, None)
        detector.detect_regions.return_value = [{
            "box": [2, 3, 22, 28],
            "detector_score": 0.9,
            "detector_class": "leaf",
        }]
        zone_id = 4
        before = get_zone(zone_id)["plants_scanned"]
        with (
            patch("backend.main.get_plant_detector", return_value=detector),
            patch("backend.main.model_readiness", return_value=(False, "four-class model unavailable")),
            patch("backend.main.disease_model_readiness", return_value=(True, None)),
            patch(
                "backend.main.predict_plantdoc_disease",
                return_value={
                    "prediction": "Disease",
                    "class_id": 0,
                    "matched_class": "Disease",
                    "confidence": 0.81,
                    "confidence_is_calibrated": False,
                    "classification_scope": "disease_only",
                    "model": "test disease model",
                },
            ),
        ):
            result = scan(FrameRequest(image=encoded_test_image(), zone_id=zone_id))

        self.assertEqual(result["status"], "DISEASE_CLASSIFIER_ONLY")
        self.assertEqual(result["classification_scope"], "disease_only")
        self.assertEqual(result["detections"][0]["prediction"], "Disease")
        self.assertEqual(result["zone_summary"]["risk"], "NOT_SCANNED")
        self.assertEqual(result["zone_summary"]["disease"], 0)
        self.assertEqual(get_zone(zone_id)["plants_scanned"], before)

    def test_missing_plant_detector_blocks_classification_and_zone_updates(self):
        detector = Mock()
        detector.readiness.return_value = (False, "detector training required")
        before = get_zone(1)["plants_scanned"]

        with (
            patch("backend.main.get_plant_detector", return_value=detector),
            patch("backend.main.model_readiness") as model_readiness,
            patch("backend.main.predict_crop") as predict_crop,
        ):
            result = scan(FrameRequest(image=encoded_test_image(), zone_id=1))

        self.assertEqual(result["status"], "DETECTION_UNAVAILABLE")
        self.assertEqual(result["message"], "detector training required")
        self.assertIsNone(result["plant_detected"])
        self.assertFalse(result["detection_available"])
        self.assertEqual(result["detections"], [])
        detector.detect_regions.assert_not_called()
        model_readiness.assert_not_called()
        predict_crop.assert_not_called()
        self.assertEqual(get_zone(1)["plants_scanned"], before)

    def test_health_reports_pipeline_not_ready_without_plant_detector(self):
        detector = Mock()
        detector.readiness.return_value = (False, "detector training required")

        with (
            patch("backend.main.get_plant_detector", return_value=detector),
            patch("backend.main.model_readiness", return_value=(True, None)),
        ):
            result = health()

        self.assertFalse(result["model_ready"])
        self.assertEqual(result["status"], "model_not_ready")
        self.assertEqual(result["message"], "detector training required")

    def test_real_detector_boxes_are_cropped_and_classified_independently(self):
        detector = Mock()
        detector.readiness.return_value = (True, None)
        detector.detect_regions.return_value = [
            {
                "box": [2.8, 3.4, 22.9, 28.5],
                "detector_score": 0.91,
                "detector_class": "plant",
            },
            {
                "box": [20, 18, 40, 40],
                "detector_score": 0.84,
                "detector_class": "leaf",
            },
        ]
        results = [
            {
                "prediction": "Healthy",
                "class_id": 0,
                "confidence": 0.88,
                "confidence_is_calibrated": False,
                "model": "test model",
            },
            {
                "prediction": "Pest",
                "class_id": 1,
                "confidence": 0.77,
                "confidence_is_calibrated": False,
                "model": "test model",
            },
        ]
        zone_id = 6
        reset_zone(zone_id)
        with (
            patch("backend.main.get_plant_detector", return_value=detector),
            patch("backend.main.model_readiness", return_value=(True, None)),
            patch("backend.main.predict_crop", side_effect=results) as predict_crop,
        ):
            result = scan(FrameRequest(image=encoded_test_image(), zone_id=zone_id))

        self.assertEqual(result["status"], "SUCCESS")
        self.assertTrue(result["plant_detected"])
        self.assertEqual(result["detections"][0]["box"], [2, 3, 20, 25])
        self.assertEqual(result["detections"][1]["box"], [20, 18, 12, 14])
        self.assertEqual(
            [call.args[0].size for call in predict_crop.call_args_list],
            [(20, 25), (12, 14)],
        )
        self.assertEqual([item["prediction"] for item in result["detections"]], ["Healthy", "Pest"])
        self.assertEqual(get_zone(zone_id)["plants_scanned"], 2)

    def test_empty_detector_result_is_the_only_no_plant_result(self):
        detector = Mock()
        detector.readiness.return_value = (True, None)
        detector.detect_regions.return_value = []
        with (
            patch("backend.main.get_plant_detector", return_value=detector),
            patch("backend.main.model_readiness", return_value=(True, None)),
            patch("backend.main.predict_crop") as predict_crop,
        ):
            result = process_frame(FrameRequest(image=encoded_test_image()))

        self.assertEqual(result["status"], "NO_PLANT_DETECTED")
        self.assertFalse(result["plant_detected"])
        self.assertTrue(result["detection_available"])
        self.assertEqual(result["detections"], [])
        predict_crop.assert_not_called()

    def test_single_image_prediction_does_not_record_zone_counts(self):
        detector = Mock()
        detector.readiness.return_value = (True, None)
        detector.detect_regions.return_value = [
            {
                "box": [2, 3, 22, 28],
                "detector_score": 0.9,
                "detector_class": "leaf",
            }
        ]
        zone_id = 5
        before = get_zone(zone_id)["plants_scanned"]
        with (
            patch("backend.main.get_plant_detector", return_value=detector),
            patch("backend.main.model_readiness", return_value=(True, None)),
            patch(
                "backend.main.predict_crop",
                return_value={
                    "prediction": "Disease",
                    "class_id": 2,
                    "confidence": 0.88,
                    "model": "test model",
                },
            ),
        ):
            result = predict(FrameRequest(image=f"data:image/png;base64,{encoded_test_image()}"))

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["detections"][0]["prediction"], "Disease")
        self.assertEqual(get_zone(zone_id)["plants_scanned"], before)

    def test_leaf_detector_result_is_not_counted_as_a_health_class(self):
        detector = Mock()
        detector.readiness.return_value = (True, None)
        detector.detect_regions.return_value = [
            {
                "box": [2, 3, 22, 28],
                "detector_score": 0.9,
                "detector_class": "leaf",
            }
        ]
        zone_id = 6
        before = get_zone(zone_id)["plants_scanned"]
        with (
            patch("backend.main.get_plant_detector", return_value=detector),
            patch("backend.main.model_readiness", return_value=(False, "training required")),
            patch("backend.main.predict_crop") as predict_crop,
        ):
            result = scan(FrameRequest(image=encoded_test_image(), zone_id=zone_id))

        self.assertEqual(result["status"], "CLASSIFICATION_UNAVAILABLE")
        self.assertEqual(result["detections"][0]["prediction"], "Leaf")
        self.assertEqual(result["zone_summary"]["plants_scanned"], before)
        predict_crop.assert_not_called()

    def test_invalid_frame_is_rejected_before_model_readiness(self):
        with patch("backend.main.get_plant_detector") as detector:
            with self.assertRaises(Exception):
                process_frame(FrameRequest(image="eA=="))
        detector.assert_not_called()

    def test_overlapping_backend_inference_returns_busy(self):
        detector = Mock()
        detector.readiness.return_value = (True, None)
        self.assertTrue(detector is not None)
        from backend.main import _inference_lock

        _inference_lock.acquire()
        try:
            with patch("backend.main.get_plant_detector", return_value=detector):
                result = process_frame(FrameRequest(image=encoded_test_image()))
        finally:
            _inference_lock.release()

        self.assertEqual(result["status"], "INFERENCE_BUSY")
        self.assertIsNone(result["plant_detected"])
        detector.readiness.assert_not_called()

    def test_reset_zone_endpoint_clears_targeted_zone(self):
        reset = reset_zone(6)
        self.assertEqual(reset["zone_id"], 6)
        self.assertEqual(reset["plants_scanned"], 0)


if __name__ == "__main__":
    unittest.main()
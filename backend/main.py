import base64
import binascii
import io
import logging
import math
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from backend.zones import zones
from quantum_ml.plant_detector import PlantDetector, PlantDetectorNotReady
from quantum_ml.predict_plantdoc_disease import (
    disease_model_readiness,
    predict_plantdoc_disease,
)
from quantum_ml.predict_quantum import model_readiness, predict_crop


logger = logging.getLogger(__name__)
app = FastAPI(title="FARM RAKSHAK API")
_plant_detector = None
_inference_lock = Lock()


class FrameRequest(BaseModel):
    image: str = Field(min_length=1)
    zone_id: int | None = Field(default=None, ge=1, le=6)


def get_plant_detector():
    global _plant_detector
    if _plant_detector is None:
        _plant_detector = PlantDetector()
    return _plant_detector


def decode_image(image_data):
    encoded_data = image_data.split(",", 1)[1] if "," in image_data else image_data
    try:
        image_bytes = base64.b64decode(encoded_data, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            return image.convert("RGB")
    except (binascii.Error, OSError, ValueError, UnidentifiedImageError) as error:
        raise HTTPException(status_code=400, detail="Image must be a valid base64-encoded image") from error


def process_frame(request, record_scan=False):
    if record_scan and request.zone_id is None:
        raise HTTPException(status_code=422, detail="zone_id is required for a farm scan")

    image = decode_image(request.image)
    if not _inference_lock.acquire(blocking=False):
        return {
            "status": "INFERENCE_BUSY",
            "message": "Another frame is being analyzed; retry after it completes",
            "plant_detected": None,
            "detection_available": None,
            "detections": [],
        }
    try:
        return _process_frame(request, image, record_scan)
    finally:
        _inference_lock.release()


def _process_frame(request, image, record_scan):
    detector = get_plant_detector()
    detector_ready, detector_reason = detector.readiness()
    if not detector_ready:
        return {
            "status": "DETECTION_UNAVAILABLE",
            "message": detector_reason,
            "plant_detected": None,
            "detection_available": False,
            "detections": [],
            "zone": f"Zone {request.zone_id}" if request.zone_id else None,
        }

    try:
        detected_regions = detector.detect_regions(image)
    except PlantDetectorNotReady as error:
        return {
            "status": "DETECTION_UNAVAILABLE",
            "message": str(error),
            "plant_detected": None,
            "detection_available": False,
            "detections": [],
            "zone": f"Zone {request.zone_id}" if request.zone_id else None,
        }
    except Exception as error:
        logger.exception("Plant detector inference failed")
        return {
            "status": "DETECTION_ERROR",
            "message": str(error),
            "plant_detected": None,
            "detection_available": False,
            "detections": [],
        }

    classifier_ready, classifier_reason = model_readiness()
    disease_ready, disease_reason = disease_model_readiness()
    disease_only = not classifier_ready and disease_ready
    classification_ready = classifier_ready or disease_only
    classification_reason = (
        "Disease-only model: healthy, pest, and non-crop classification are unavailable"
        if disease_only
        else disease_reason
        if disease_reason and disease_reason != "PlantDoc disease model has not been trained"
        else classifier_reason
    )
    detections = []
    image_width, image_height = image.size
    for region in detected_regions:
        if not isinstance(region, dict):
            logger.warning("Ignoring invalid plant detector result: %s", region)
            return {
                "status": "DETECTION_ERROR",
                "message": "Plant detector returned a malformed detection",
                "plant_detected": None,
                "detection_available": False,
                "detections": [],
            }
        coordinates = region.get("box")
        detector_score = region.get("detector_score")
        try:
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
                raise ValueError("A bounding box must contain four coordinates")
            coordinates = [float(value) for value in coordinates]
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError("Bounding box coordinates must be finite")
            if (
                not isinstance(detector_score, (int, float))
                or not math.isfinite(detector_score)
                or not 0.0 <= detector_score <= 1.0
            ):
                raise ValueError("Detector score must be between 0 and 1")
        except (TypeError, ValueError, OverflowError) as error:
            logger.warning("Ignoring invalid plant detector result: %s", error)
            return {
                "status": "DETECTION_ERROR",
                "message": f"Plant detector returned an invalid detection: {error}",
                "plant_detected": None,
                "detection_available": False,
                "detections": [],
            }

        x1, y1, x2, y2 = coordinates
        left = max(0, min(image_width, int(x1)))
        top = max(0, min(image_height, int(y1)))
        right = max(0, min(image_width, int(x2)))
        bottom = max(0, min(image_height, int(y2)))
        if right <= left or bottom <= top:
            logger.warning("Ignoring invalid plant detector box: %s", region["box"])
            return {
                "status": "DETECTION_ERROR",
                "message": "Plant detector returned a box outside the camera frame",
                "plant_detected": None,
                "detection_available": False,
                "detections": [],
            }

        if classifier_ready or disease_only:
            cropped_image = image.crop((left, top, right, bottom))
            try:
                result = (
                    predict_crop(cropped_image)
                    if classifier_ready
                    else predict_plantdoc_disease(cropped_image)
                )
            except Exception as error:
                logger.exception("Plant crop classification failed")
                return {
                    "status": "INFERENCE_ERROR",
                    "message": str(error),
                    "plant_detected": True,
                    "detection_available": True,
                    "detections": detections,
                    "zone": f"Zone {request.zone_id}" if request.zone_id else None,
                }
        else:
            result = {
                "prediction": "Leaf",
                "class_id": None,
                "confidence": None,
                "confidence_is_calibrated": False,
                "model": "PlantDoc SSDLite leaf detector",
            }
        detections.append({
            "box": [left, top, right - left, bottom - top],
            "prediction": result["prediction"],
            "class_id": result["class_id"],
            "confidence": result["confidence"],
            "confidence_is_calibrated": result.get("confidence_is_calibrated", False),
            "detector_score": detector_score,
            "detector_class": region["detector_class"],
            "model": result["model"],
        })

    valid_predictions = [
        detection["prediction"]
        for detection in detections
        if detection["prediction"] in ("Healthy", "Pest", "Disease")
    ]
    response = {
        "status": (
            "NO_PLANT_DETECTED" if not detections
            else "SUCCESS" if classifier_ready
            else "DISEASE_CLASSIFIER_ONLY" if disease_only
            else "CLASSIFICATION_UNAVAILABLE"
        ),
        "message": (
            "No plant detected" if not detections
            else None if classification_ready
            else classification_reason
        ),
        "classification_scope": (
            "four_class" if classifier_ready
            else "disease_only" if disease_only
            else None
        ),
        "plant_detected": bool(detections),
        "detection_available": True,
        "detections": detections,
        "zone": f"Zone {request.zone_id}" if request.zone_id else None,
    }

    if record_scan:
        summary = zones.record(
            request.zone_id,
            valid_predictions if classifier_ready else [],
        )
        response["zone_summary"] = summary
        response["risk"] = summary["risk"]
    if record_scan and disease_only:
        response["message"] = classification_reason
    return response


@app.get("/health")
def health():
    detector_ready, detector_reason = get_plant_detector().readiness()
    model_ready, model_reason = model_readiness()
    disease_ready, disease_reason = disease_model_readiness()
    ready = detector_ready and model_ready
    reason = (
        None if ready
        else detector_reason if not detector_ready
        else "Disease-only model is ready; healthy, pest, and non-crop classification are unavailable"
        if disease_ready
        else disease_reason
        if disease_reason != "PlantDoc disease model has not been trained"
        else model_reason or disease_reason
    )
    return {
        "status": "ready" if ready else "partial_ready" if detector_ready and disease_ready else "model_not_ready",
        "model_ready": ready,
        "detector_ready": detector_ready,
        "classifier_ready": model_ready,
        "disease_classifier_ready": disease_ready,
        "detector_message": detector_reason,
        "classifier_message": model_reason,
        "disease_classifier_message": disease_reason,
        "message": reason,
    }


@app.post("/predict")
def predict(request: FrameRequest):
    return process_frame(request)


@app.post("/scan")
def scan(request: FrameRequest):
    return process_frame(request, record_scan=True)


@app.post("/predict_frame")
def predict_frame(request: FrameRequest):
    return process_frame(request, record_scan=request.zone_id is not None)


@app.get("/zones")
def get_zones():
    return {"zones": zones.all()}


@app.get("/zones/{zone_id}")
def get_zone(zone_id: int):
    try:
        return zones.get(zone_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Zone not found") from error


@app.post("/zones/{zone_id}/reset")
def reset_zone(zone_id: int):
    try:
        return zones.reset(zone_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Zone not found") from error


frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
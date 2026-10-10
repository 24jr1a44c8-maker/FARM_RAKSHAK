import logging
import os
from pathlib import Path
from threading import Lock

from PIL import Image


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "plant_detector" / "best.pt"
V2_MODEL_PATH = PROJECT_ROOT / "models" / "plant_detector_v2" / "torch_ssdlite" / "leaf_detector.pth"
DEFAULT_PLANT_CLASSES = {"plant", "leaf", "crop", "potted plant"}


class PlantDetectorNotReady(RuntimeError):
    pass


class PlantDetector:
    """Loads a configured YOLO model trained to localize plants or leaves."""

    def __init__(self):
        configured_path = os.getenv("PLANT_DETECTOR_MODEL")
        configured_confidence = os.getenv("PLANT_DETECTOR_CONFIDENCE")
        self._confidence_is_configured = configured_confidence is not None
        if configured_path:
            self.model_path = Path(configured_path)
        elif V2_MODEL_PATH.is_file():
            self.model_path = V2_MODEL_PATH
        else:
            self.model_path = DEFAULT_MODEL_PATH
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path
        default_threshold = "0.35" if self.model_path.suffix.casefold() == ".pth" else "0.25"
        try:
            self.confidence_threshold = float(
                configured_confidence if configured_confidence is not None else default_threshold
            )
        except ValueError:
            self.confidence_threshold = None
        configured_classes = os.getenv("PLANT_DETECTOR_CLASSES")
        self.plant_classes = (
            {name.strip().casefold() for name in configured_classes.split(",") if name.strip()}
            if configured_classes
            else DEFAULT_PLANT_CLASSES
        )
        self._model = None
        self._load_error = None
        self._load_lock = Lock()

    def _load_model(self):
        if self.model_path.suffix.casefold() == ".pth":
            import torch
            from torchvision.models.detection import ssdlite320_mobilenet_v3_large

            if "leaf" not in self.plant_classes:
                raise RuntimeError("Torchvision PlantDoc detector supports only the leaf class")
            checkpoint = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=True,
            )
            if checkpoint.get("class_names") != ["background", "leaf"]:
                raise RuntimeError("Torchvision detector checkpoint must use class order: background, leaf")
            checkpoint_threshold = checkpoint.get("confidence_threshold")
            if not self._confidence_is_configured and checkpoint_threshold is not None:
                if (
                    not isinstance(checkpoint_threshold, (int, float))
                    or not 0.0 <= checkpoint_threshold <= 1.0
                ):
                    raise RuntimeError("Torchvision detector confidence threshold must be between 0 and 1")
                self.confidence_threshold = float(checkpoint_threshold)
            model = ssdlite320_mobilenet_v3_large(
                weights=None,
                weights_backbone=None,
                num_classes=2,
            )
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            return model.eval()

        from ultralytics import YOLO

        model = YOLO(str(self.model_path))
        names = model.names
        class_names = names.values() if isinstance(names, dict) else names
        if not any(str(name).strip().casefold() in self.plant_classes for name in class_names):
            raise RuntimeError(
                "Configured detector has no plant/leaf class. Set "
                "PLANT_DETECTOR_CLASSES to match its trained class names."
            )
        return model

    def readiness(self):
        with self._load_lock:
            if self._model is not None:
                return True, None
            if self._load_error is not None:
                return False, self._load_error
            if self.confidence_threshold is None or not 0.0 <= self.confidence_threshold <= 1.0:
                self._load_error = "PLANT_DETECTOR_CONFIDENCE must be between 0 and 1"
                return False, self._load_error
            if not self.plant_classes:
                self._load_error = "PLANT_DETECTOR_CLASSES must contain at least one class name"
                return False, self._load_error
            if not self.model_path.is_file():
                return False, (
                    f"Plant/leaf detector weights are missing: {self.model_path}. "
                    "Train the separate PlantDoc leaf detector or configure weights with "
                    "PLANT_DETECTOR_MODEL."
                )
            try:
                self._model = self._load_model()
                if self.model_path.suffix.casefold() == ".pth":
                    return True, None
            except Exception as error:
                logger.exception("Could not load configured plant detector")
                self._load_error = f"Plant detector failed to load: {error}"
                return False, self._load_error
            return True, None

    def detect_regions(self, image: Image.Image):
        ready, message = self.readiness()
        if not ready:
            raise PlantDetectorNotReady(message)

        if self.model_path.suffix.casefold() == ".pth":
            import torch
            from torchvision.transforms.functional import pil_to_tensor

            tensor = pil_to_tensor(image.convert("RGB")).float().div_(255)
            with torch.inference_mode():
                result = self._model([tensor])[0]
            detections = []
            for box, class_id, score in zip(
                result["boxes"].detach().cpu().tolist(),
                result["labels"].detach().cpu().tolist(),
                result["scores"].detach().cpu().tolist(),
            ):
                if int(class_id) == 1 and float(score) >= self.confidence_threshold:
                    detections.append({
                        "box": [float(value) for value in box],
                        "detector_score": float(score),
                        "detector_class": "leaf",
                    })
            return detections

        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return []

        names = result.names
        class_names = list(names.values()) if isinstance(names, dict) else names
        detections = []
        coordinates = boxes.xyxy.detach().cpu().tolist()
        class_ids = boxes.cls.detach().cpu().tolist()
        scores = boxes.conf.detach().cpu().tolist()
        if len(coordinates) != len(class_ids) or len(class_ids) != len(scores):
            raise RuntimeError("Plant detector returned inconsistent box, class, and score counts")
        for coordinates_for_box, class_id, score in zip(coordinates, class_ids, scores):
            class_name = str(
                names[int(class_id)] if isinstance(names, dict) else class_names[int(class_id)]
            ).strip()
            if class_name.casefold() not in self.plant_classes:
                continue
            x1, y1, x2, y2 = (float(value) for value in coordinates_for_box)
            detections.append({
                "box": [x1, y1, x2, y2],
                "detector_score": float(score),
                "detector_class": class_name,
            })
        return detections

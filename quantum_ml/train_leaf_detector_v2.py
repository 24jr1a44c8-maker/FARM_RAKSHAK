import argparse
import json
import math
import os
import random
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.transforms.functional import pil_to_tensor


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "models" / "plant_detector_v2" / "dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "plant_detector_v2" / "torch_ssdlite"
CLASS_NAMES = ("background", "leaf")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CHECKPOINT_NAME = "leaf_detector.pth"
METRICS_NAME = "evaluation.json"


class YoloLeafDataset(Dataset):
    def __init__(self, dataset_dir, split_name, max_images=None):
        self.image_dir = Path(dataset_dir) / "images" / split_name
        self.label_dir = Path(dataset_dir) / "labels" / split_name
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(f"Detector image/label split is missing: {split_dir}")
        self.image_paths = sorted(
            path for path in self.image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise ValueError(f"No detector images found in {self.image_dir}")
        if max_images is not None and max_images > 0 and len(self.image_paths) > max_images:
            selected = np.linspace(0, len(self.image_paths) - 1, max_images, dtype=int)
            self.image_paths = [self.image_paths[index] for index in selected]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Detector labels are missing for {image_path.name}")

        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            image_tensor = pil_to_tensor(image).float().div_(255)

        boxes = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5 or fields[0] != "0":
                raise ValueError(f"Expected one leaf class and four coordinates in {label_path}:{line_number}")
            try:
                center_x, center_y, box_width, box_height = map(float, fields[1:])
            except ValueError as error:
                raise ValueError(f"Invalid numeric box in {label_path}:{line_number}") from error
            values = (center_x, center_y, box_width, box_height)
            if (
                not all(math.isfinite(value) for value in values)
                or not 0 <= center_x <= 1
                or not 0 <= center_y <= 1
                or not 0 < box_width <= 1
                or not 0 < box_height <= 1
            ):
                raise ValueError(f"Invalid normalized box in {label_path}:{line_number}")
            x1 = max(0.0, (center_x - box_width / 2) * width)
            y1 = max(0.0, (center_y - box_height / 2) * height)
            x2 = min(float(width), (center_x + box_width / 2) * width)
            y2 = min(float(height), (center_y + box_height / 2) * height)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Empty leaf box in {label_path}:{line_number}")
            boxes.append((x1, y1, x2, y2))
        if not boxes:
            raise ValueError(f"No leaf boxes found in {label_path}")

        box_tensor = torch.tensor(boxes, dtype=torch.float32)
        target = {
            "boxes": box_tensor,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return image_tensor, target


def _collate(batch):
    return tuple(zip(*batch))


def _cached_coco_weights():
    filename = Path(urlparse(SSDLite320_MobileNet_V3_Large_Weights.DEFAULT.url).path).name
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "The Torchvision SSDLite COCO weights are not cached locally. "
            f"Expected: {checkpoint_path}. No weights or dataset files were downloaded."
        )
    return torch.load(checkpoint_path, map_location="cpu", weights_only=True)


def _build_model(device, pretrained_state=None):
    model = ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        num_classes=len(CLASS_NAMES),
    )
    if pretrained_state is not None:
        model_state = model.state_dict()
        compatible = {
            name: value
            for name, value in pretrained_state.items()
            if name in model_state and value.shape == model_state[name].shape
        }
        if not compatible:
            raise RuntimeError("Cached SSDLite weights do not match the installed Torchvision model")
        model.load_state_dict(compatible, strict=False)
        model._pretrained_parameter_count = len(compatible)
    return model.to(device)


def _box_iou(box, boxes):
    left = torch.maximum(box[0], boxes[:, 0])
    top = torch.maximum(box[1], boxes[:, 1])
    right = torch.minimum(box[2], boxes[:, 2])
    bottom = torch.minimum(box[3], boxes[:, 3])
    intersection = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
    box_area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    return intersection / (box_area + areas - intersection).clamp(min=1e-8)


def _predict_dataset(model, dataset, device):
    model.eval()
    predictions = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            prediction = model([image.to(device)])[0]
            predictions.append({
                "boxes": prediction["boxes"].detach().cpu(),
                "labels": prediction["labels"].detach().cpu(),
                "scores": prediction["scores"].detach().cpu(),
                "ground_truth": target["boxes"],
            })
    return predictions


def _measure_predictions(predictions, score_threshold, iou_threshold=0.5):
    true_positives = false_positives = false_negatives = 0
    for prediction in predictions:
        predicted_boxes = prediction["boxes"]
        predicted_labels = prediction["labels"]
        scores = prediction["scores"]
        keep = (predicted_labels == 1) & (scores >= score_threshold)
        predicted_boxes = predicted_boxes[keep]
        scores = scores[keep]
        if scores.numel():
            order = scores.argsort(descending=True)
            predicted_boxes = predicted_boxes[order]

        ground_truth = prediction["ground_truth"]
        matched = torch.zeros(len(ground_truth), dtype=torch.bool)
        image_true_positives = 0
        for predicted_box in predicted_boxes:
            overlaps = _box_iou(predicted_box, ground_truth)
            overlaps[matched] = -1
            best_overlap, best_index = overlaps.max(dim=0)
            if best_overlap >= iou_threshold:
                matched[best_index] = True
                image_true_positives += 1
            else:
                false_positives += 1
        true_positives += image_true_positives
        false_negatives += len(ground_truth) - image_true_positives

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "images": len(predictions),
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_model(model, dataset, device, score_threshold=0.25, iou_threshold=0.5):
    return _measure_predictions(
        _predict_dataset(model, dataset, device),
        score_threshold,
        iou_threshold,
    )


def train_detector(
    dataset_dir=DATASET_DIR,
    model_dir=MODEL_DIR,
    epochs=3,
    batch_size=4,
    max_train_images=256,
    max_validation_images=32,
    max_test_images=32,
):
    dataset_dir = Path(dataset_dir)
    model_dir = Path(model_dir)
    checkpoint_path = model_dir / CHECKPOINT_NAME
    metrics_path = model_dir / METRICS_NAME
    if checkpoint_path.exists() or metrics_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing detector artifacts in {model_dir}"
        )
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must both be positive")

    full_split_counts = {
        name: len(YoloLeafDataset(dataset_dir, name))
        for name in ("train", "validation", "test")
    }
    train_set = YoloLeafDataset(dataset_dir, "train", max_train_images or None)
    validation_set = YoloLeafDataset(
        dataset_dir, "validation", max_validation_images or None
    )
    test_set = YoloLeafDataset(dataset_dir, "test", max_test_images or None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_state = _cached_coco_weights()
    model = _build_model(device, pretrained_state)
    del pretrained_state
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
    )
    optimizer = torch.optim.SGD(
        model.head.parameters(),
        lr=0.001,
        momentum=0.9,
        weight_decay=0.0005,
    )
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    history = []
    for epoch in range(epochs):
        model.train()
        model.backbone.eval()
        epoch_loss = 0.0
        for batch_index, (images, targets) in enumerate(train_loader, 1):
            loss_values = model(
                [image.to(device) for image in images],
                [{key: value.to(device) for key, value in target.items()} for target in targets],
            )
            loss = sum(loss_values.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}, batch {batch_index}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            if batch_index % 50 == 0 or batch_index == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"batch {batch_index}/{len(train_loader)} "
                    f"loss={float(loss.detach().cpu()):.4f}"
                )
        mean_loss = epoch_loss / len(train_loader)
        history.append({"epoch": epoch + 1, "mean_loss": mean_loss})
        print(f"Epoch {epoch + 1}/{epochs} mean loss={mean_loss:.4f}")

    validation_predictions = _predict_dataset(model, validation_set, device)
    threshold_sweep = {
        threshold: _measure_predictions(validation_predictions, threshold)
        for threshold in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
    }
    calibrated_threshold = max(
        threshold_sweep,
        key=lambda threshold: threshold_sweep[threshold]["f1"],
    )
    validation_metrics = threshold_sweep[calibrated_threshold]
    test_metrics = _measure_predictions(
        _predict_dataset(model, test_set, device),
        calibrated_threshold,
    )
    model.eval()
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "class_names": list(CLASS_NAMES),
        "architecture": "torchvision ssdlite320_mobilenet_v3_large",
        "model_version": 1,
        "input_color": "RGB",
        "confidence_threshold": float(calibrated_threshold),
    }
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".tmp", dir=model_dir, delete=False
    ) as temporary:
        checkpoint_temp = Path(temporary.name)
    try:
        torch.save(checkpoint, checkpoint_temp)
        os.replace(checkpoint_temp, checkpoint_path)
    finally:
        checkpoint_temp.unlink(missing_ok=True)

    metrics = {
        "task": "single-class leaf object detection",
        "class_names": list(CLASS_NAMES),
        "dataset_dir": str(dataset_dir.resolve()),
        "pretrained_initialization": "cached Torchvision SSDLite COCO checkpoint",
        "pretrained_tensors_loaded": getattr(model, "_pretrained_parameter_count", None),
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "training_images": len(train_set),
        "available_split_images": full_split_counts,
        "evaluation_images": {
            "validation": len(validation_set),
            "held_out_test": len(test_set),
        },
        "frozen_backbone": True,
        "confidence_threshold_selected_on_validation": calibrated_threshold,
        "validation_threshold_sweep": threshold_sweep,
        "validation": validation_metrics,
        "held_out_test": test_metrics,
        "training_history": history,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "negative_images_available": False,
        "pest_detection_supported": False,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved leaf detector to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a separate SSDLite leaf detector on prepared PlantDoc boxes."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-images", type=int, default=256)
    parser.add_argument("--max-validation-images", type=int, default=32)
    parser.add_argument("--max-test-images", type=int, default=32)
    args = parser.parse_args()
    train_detector(
        dataset_dir=args.dataset_dir,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_train_images=args.max_train_images,
        max_validation_images=args.max_validation_images,
        max_test_images=args.max_test_images,
    )


if __name__ == "__main__":
    main()

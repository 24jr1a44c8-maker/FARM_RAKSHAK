import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from quantum_ml.cnn_features import CNNFeatureExtractor
from quantum_ml.preprocessing import FeatureReducer
from quantum_ml.predict_quantum import MODEL_METADATA_VERSION
from quantum_ml.vqc_model import QuantumVQCModel


CLASS_NAMES = ["Healthy", "Pest", "Disease", "Non-Crop"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models" / "quantum_model"


def load_split(split_name):
    images = []
    labels = []
    counts = {}
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = DATASET_DIR / split_name / class_name.lower().replace("-", "_")
        paths = [
            class_dir / filename
            for filename in sorted(os.listdir(class_dir))
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        ] if os.path.isdir(class_dir) else []
        counts[class_name] = len(paths)
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
            labels.append(label)
    return images, np.asarray(labels, dtype=int), counts


def extract_features(images, extractor):
    if not images:
        return np.empty((0, 1280), dtype=np.float32)
    return np.asarray([extractor.extract_features(image) for image in images])


def report_metrics(name, expected, predicted):
    print(f"\n{name} evaluation ({len(expected)} held-out samples)")
    print(f"Accuracy: {accuracy_score(expected, predicted):.4f}")
    print(f"Macro F1: {f1_score(expected, predicted, labels=range(4), average='macro', zero_division=0):.4f}")
    print("Per-class precision/recall/F1:")
    print(classification_report(expected, predicted, labels=range(4), target_names=CLASS_NAMES, zero_division=0))
    print("Confusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(expected, predicted, labels=range(4)))


def train_pipeline():
    print("Loading labeled images from dataset/train/<class>")
    train_images, y_train, train_counts = load_split("train")
    missing_classes = [name for name, count in train_counts.items() if count == 0]
    if missing_classes:
        raise RuntimeError(
            "Training requires real labeled images for all four classes. "
            f"Missing: {', '.join(missing_classes)}"
        )
    if len(train_images) < 4:
        raise RuntimeError("At least four real training images are required to fit four PCA components")

    validation_images, _, validation_counts = load_split("validation")
    test_images, y_test, test_counts = load_split("test")

    cnn = CNNFeatureExtractor()
    train_features = extract_features(train_images, cnn)
    reducer = FeatureReducer(n_components=4)
    reducer.fit(train_features)
    reduced_train = reducer.reduce(train_features)

    vqc_model = QuantumVQCModel(num_features=4, model_path=str(MODEL_DIR / "vqc.model"))
    vqc_model.build_and_train(reduced_train, np.eye(4)[y_train])

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact_names = ("scaler.pkl", "pca.pkl", "vqc.model")
    artifacts = {}
    for filename in artifact_names:
        digest = hashlib.sha256()
        with open(MODEL_DIR / filename, "rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
        artifacts[filename] = digest.hexdigest()

    metadata = {
        "model_metadata_version": MODEL_METADATA_VERSION,
        "class_names": CLASS_NAMES,
        "training_samples": len(train_images),
        "training_counts": train_counts,
        "artifacts": artifacts,
        "validation_counts": validation_counts,
        "test_counts": test_counts,
        "feature_extractor": "torchvision MobileNetV2 ImageNet weights",
        "reduction": "StandardScaler + 4-component PCA",
        "classifier": "Qiskit Machine Learning VQC with StatevectorSampler",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = MODEL_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    if test_images:
        test_features = extract_features(test_images, cnn)
        reduced_test = reducer.reduce(test_features)
        quantum_probabilities = vqc_model.vqc.predict_proba(reduced_test)
        quantum_predictions = np.asarray(quantum_probabilities).argmax(axis=1)
        report_metrics("CNN + PCA + Qiskit VQC", y_test, quantum_predictions)

        baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
        baseline.fit(reduced_train, y_train)
        report_metrics("CNN + PCA + LogisticRegression baseline", y_test, baseline.predict(reduced_test))
    else:
        print("No test images found; metrics and model comparison were not generated.")

    if validation_images:
        print(f"Validation images available: {len(validation_images)} (kept separate from fitting)")
    print(f"Trained model metadata saved to {metadata_path}")


if __name__ == "__main__":
    train_pipeline()
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from quantum_ml.cnn_features import CNNFeatureExtractor
from quantum_ml.predict_quantum import MODEL_METADATA_VERSION
from quantum_ml.preprocessing import FeatureReducer
from quantum_ml.train_quantum import CLASS_NAMES, IMAGE_EXTENSIONS, extract_features
from quantum_ml.vqc_model import QuantumVQCModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "classifier_v2"
MODEL_DIR = PROJECT_ROOT / "models" / "quantum_model_v2"


def load_split(split_name, dataset_dir=DATASET_DIR):
    images = []
    labels = []
    counts = {}
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = dataset_dir / split_name / class_name.lower().replace("-", "_")
        paths = [
            class_dir / filename
            for filename in sorted(os.listdir(class_dir))
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS
        ] if class_dir.is_dir() else []
        counts[class_name] = len(paths)
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
            labels.append(label)
    return images, np.asarray(labels, dtype=int), counts


def _report_metrics(name, expected, predicted):
    print(f"\n{name} evaluation ({len(expected)} held-out samples)")
    print(f"Accuracy: {accuracy_score(expected, predicted):.4f}")
    print(f"Macro F1: {f1_score(expected, predicted, labels=range(4), average='macro', zero_division=0):.4f}")
    print("Per-class precision/recall/F1:")
    print(classification_report(
        expected,
        predicted,
        labels=range(4),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    print("Confusion matrix (actual rows, predicted columns):")
    print(confusion_matrix(expected, predicted, labels=range(4)))


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_pipeline(dataset_dir=DATASET_DIR, model_dir=MODEL_DIR):
    dataset_dir = Path(dataset_dir)
    model_dir = Path(model_dir)
    train_images, y_train, train_counts = load_split("train", dataset_dir)
    missing_classes = [name for name, count in train_counts.items() if count == 0]
    if missing_classes:
        raise RuntimeError(
            "Training requires real labeled images for all four classes. "
            f"Missing: {', '.join(missing_classes)}"
        )
    if len(train_images) < 4:
        raise RuntimeError("At least four real training images are required to fit four PCA components")
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite files in the new-model directory: {model_dir}"
        )

    validation_images, _, validation_counts = load_split("validation", dataset_dir)
    test_images, y_test, test_counts = load_split("test", dataset_dir)

    cnn = CNNFeatureExtractor()
    train_features = extract_features(train_images, cnn)
    reducer = FeatureReducer(n_components=4, model_dir=str(model_dir))
    reducer.fit(train_features)
    reduced_train = reducer.reduce(train_features)

    vqc_model = QuantumVQCModel(
        num_features=4,
        model_path=str(model_dir / "vqc.model"),
    )
    vqc_model.build_and_train(reduced_train, np.eye(4)[y_train])

    artifact_names = ("scaler.pkl", "pca.pkl", "vqc.model")
    artifacts = {filename: _sha256(model_dir / filename) for filename in artifact_names}
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
    metadata_path = model_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if test_images:
        test_features = extract_features(test_images, cnn)
        reduced_test = reducer.reduce(test_features)
        probabilities = np.asarray(vqc_model.vqc.predict_proba(reduced_test))
        predictions = probabilities.argmax(axis=1)
        _report_metrics("CNN + PCA + Qiskit VQC", y_test, predictions)

        baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
        baseline.fit(reduced_train, y_train)
        _report_metrics(
            "CNN + PCA + LogisticRegression baseline",
            y_test,
            baseline.predict(reduced_test),
        )
    else:
        print("No test images found; metrics and model comparison were not generated.")

    if validation_images:
        print(f"Validation images available: {len(validation_images)} (kept separate from fitting)")
    print(f"New model metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a separate CNN + Qiskit four-class model without overwriting the old model."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    train_pipeline(dataset_dir=args.dataset_dir, model_dir=args.model_dir)


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from quantum_ml.cnn_features import CNNFeatureExtractor
from quantum_ml.predict_quantum import _sha256
from quantum_ml.preprocessing import FeatureReducer
from quantum_ml.train_quantum import IMAGE_EXTENSIONS, extract_features
from quantum_ml.vqc_model import QuantumVQCModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "plantdoc_disease_v2"
MODEL_DIR = PROJECT_ROOT / "models" / "plantdoc_disease_v2"
CLASS_NAMES = ("Disease", "Unspecified Leaf")
CLASS_DIRECTORIES = ("disease", "unspecified_leaf")
METADATA_VERSION = 1


def load_split(split_name, dataset_dir=DATASET_DIR, max_per_class=0):
    images = []
    labels = []
    counts = {}
    for label, directory_name in enumerate(CLASS_DIRECTORIES):
        class_dir = Path(dataset_dir) / split_name / directory_name
        paths = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ) if class_dir.is_dir() else []
        if max_per_class:
            paths = sorted(paths, key=lambda path: hashlib.sha256(path.name.encode()).hexdigest())
            paths = paths[:max_per_class]
        counts[CLASS_NAMES[label]] = len(paths)
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
            labels.append(label)
    return images, np.asarray(labels, dtype=int), counts


def _report_metrics(name, expected, predicted):
    report = classification_report(
        expected,
        predicted,
        labels=range(2),
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    metrics = {
        "accuracy": float(accuracy_score(expected, predicted)),
        "macro_f1": float(
            f1_score(expected, predicted, labels=range(2), average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(expected, predicted, labels=range(2)).tolist(),
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        },
    }
    print(f"\n{name} evaluation ({len(expected)} held-out crops)")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("Per-class precision/recall/F1:")
    print(classification_report(
        expected,
        predicted,
        labels=range(2),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))
    print("Confusion matrix (actual rows, predicted columns):")
    print(metrics["confusion_matrix"])
    return metrics


def train_pipeline(dataset_dir=DATASET_DIR, model_dir=MODEL_DIR, max_train_per_class=512):
    dataset_dir = Path(dataset_dir)
    model_dir = Path(model_dir)
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite model artifacts: {model_dir}")

    train_images, y_train, train_counts = load_split(
        "train", dataset_dir, max_train_per_class
    )
    validation_images, y_validation, validation_counts = load_split("validation", dataset_dir)
    test_images, y_test, test_counts = load_split("test", dataset_dir)
    for split_name, counts in (
        ("training", train_counts),
        ("validation", validation_counts),
        ("held-out test", test_counts),
    ):
        missing = [name for name, count in counts.items() if count == 0]
        if missing:
            raise RuntimeError(
                f"Both PlantDoc labels need images in the {split_name} split. "
                f"Missing: {', '.join(missing)}"
            )
    if len(train_images) < 4:
        raise RuntimeError("At least four training crops are required to fit PCA")

    cnn = CNNFeatureExtractor()
    reducer = FeatureReducer(n_components=4, model_dir=str(model_dir))
    train_features = extract_features(train_images, cnn)
    reducer.fit(train_features)
    reduced_train = reducer.reduce(train_features)

    vqc_model = QuantumVQCModel(
        num_features=4,
        model_path=str(model_dir / "vqc.model"),
    )
    vqc_model.build_and_train(reduced_train, np.eye(2)[y_train])

    validation_probabilities = np.asarray(
        vqc_model.vqc.predict_proba(reducer.reduce(extract_features(validation_images, cnn))),
        dtype=float,
    )
    if validation_probabilities.shape != (len(validation_images), 2):
        raise RuntimeError("Disease VQC did not return both configured class scores")
    validation_metrics = _report_metrics(
        "CNN + PCA + Qiskit VQC validation",
        y_validation,
        validation_probabilities.argmax(axis=1),
    )

    test_probabilities = np.asarray(
        vqc_model.vqc.predict_proba(reducer.reduce(extract_features(test_images, cnn))),
        dtype=float,
    )
    if test_probabilities.shape != (len(test_images), 2):
        raise RuntimeError("Disease VQC did not return both configured class scores")
    test_metrics = _report_metrics(
        "CNN + PCA + Qiskit VQC held-out test",
        y_test,
        test_probabilities.argmax(axis=1),
    )

    artifact_names = ("scaler.pkl", "pca.pkl", "vqc.model")
    metadata = {
        "model_metadata_version": METADATA_VERSION,
        "class_names": list(CLASS_NAMES),
        "training_samples": len(train_images),
        "training_counts": train_counts,
        "validation_counts": validation_counts,
        "test_counts": test_counts,
        "evaluation": {
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "artifacts": {
            filename: _sha256(model_dir / filename) for filename in artifact_names
        },
        "feature_extractor": "torchvision MobileNetV2 ImageNet weights",
        "reduction": "StandardScaler + 4-component PCA",
        "classifier": "Qiskit Machine Learning VQC with StatevectorSampler",
        "dataset": "PlantDoc explicit disease categories and unspecified leaf categories",
        "disease_labels_are_pest_labels": False,
        "unspecified_leaf_means_healthy": False,
        "deployment_status": "research_only",
        "inference_approved": False,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = model_dir / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSeparate PlantDoc disease model saved to {model_dir}")
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Train a separate PlantDoc disease-match VQC; it does not predict healthy or pests."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--max-train-per-class", type=int, default=512)
    args = parser.parse_args()
    if args.max_train_per_class < 0:
        parser.error("--max-train-per-class cannot be negative")
    train_pipeline(args.dataset_dir, args.model_dir, args.max_train_per_class)


if __name__ == "__main__":
    main()

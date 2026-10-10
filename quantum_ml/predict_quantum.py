import json
import hashlib
import os
from pathlib import Path

import numpy as np

from PIL import Image
from quantum_ml.cnn_features import CNNFeatureExtractor
from quantum_ml.preprocessing import FeatureReducer
from quantum_ml.vqc_model import QuantumVQCModel

# Cache models globally so they load once, not per-frame
_cnn = None
_reducer = None
_vqc = None
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "quantum_model"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
try:
    CONFIDENCE_THRESHOLD = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.55"))
except ValueError:
    CONFIDENCE_THRESHOLD = None
LABELS_4 = {0: "Healthy", 1: "Pest", 2: "Disease", 3: "Non-Crop"}
MODEL_METADATA_VERSION = 2


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_readiness():
    if CONFIDENCE_THRESHOLD is None or not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
        return False, "PREDICTION_CONFIDENCE_THRESHOLD must be between 0 and 1"
    if not os.path.exists(METADATA_PATH):
        return False, "Quantum model has not been trained from labeled data"
    try:
        with open(METADATA_PATH, encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if not isinstance(metadata, dict):
            return False, "Quantum model metadata is missing or invalid"
        training_counts = metadata.get("training_counts", {})
        artifacts = metadata.get("artifacts", {})
        if not isinstance(training_counts, dict) or not isinstance(artifacts, dict):
            return False, "Quantum model metadata is missing or invalid"
        classes_valid = all(
            type(training_counts.get(name)) is int and training_counts[name] > 0
            for name in LABELS_4.values()
        )
        samples_valid = (
            type(metadata.get("training_samples")) is int
            and metadata["training_samples"] == sum(training_counts.values())
        )
        artifacts_valid = all(
            artifacts.get(filename) == _sha256(MODEL_DIR / filename)
            for filename in ("scaler.pkl", "pca.pkl", "vqc.model")
            if (MODEL_DIR / filename).is_file()
        )
        all_artifacts_present = all(
            (MODEL_DIR / filename).is_file()
            for filename in ("scaler.pkl", "pca.pkl", "vqc.model")
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return False, "Quantum model metadata is missing or invalid"
    class_names = metadata.get("class_names")
    if (
        metadata.get("model_metadata_version") != MODEL_METADATA_VERSION
        or class_names != list(LABELS_4.values())
        or not samples_valid
        or not classes_valid
        or not all_artifacts_present
        or not artifacts_valid
    ):
        return False, "Quantum model metadata or trained artifacts do not match the required training schema"
    return True, None

def _get_models():
    global _cnn, _reducer, _vqc
    if _cnn is None:
        _cnn = CNNFeatureExtractor()
    if _reducer is None:
        _reducer = FeatureReducer(n_components=4, model_dir=str(MODEL_DIR))
    if _vqc is None:
        _vqc = QuantumVQCModel(
            num_features=4,
            model_path=str(MODEL_DIR / "vqc.model"),
        )
    return _cnn, _reducer, _vqc

def predict_crop(image: Image.Image):
    """
    Full hybrid Classical + Quantum prediction pipeline:
    
    Image → CNN Feature Extraction → StandardScaler → PCA → Quantum Encoding → VQC → Prediction
    """
    ready, message = model_readiness()
    if not ready:
        raise RuntimeError(message)

    cnn, reducer, vqc = _get_models()
    
    # 1. CNN Feature Extraction (MobileNetV2 → 1280-dim vector)
    features = cnn.extract_features(image)
    
    # 2. StandardScaler + PCA (1280 → 4 features)
    reduced = reducer.reduce(features)

    # 3. Quantum Encoding + VQC Prediction (Qiskit circuit execution)
    outputs = vqc.vqc.predict_proba(reduced)
    probabilities = np.asarray(outputs, dtype=float)
    if probabilities.shape != (1, len(LABELS_4)):
        raise RuntimeError("Quantum model did not return scores for all four required classes")
    probabilities = probabilities[0]
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-3)
    ):
        raise RuntimeError("Quantum model returned an invalid score distribution")
    class_id = int(probabilities.argmax())
    confidence = float(probabilities[class_id])

    class_labels = LABELS_4
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "prediction": "UNCERTAIN",
            "class_id": class_id,
            "confidence": round(confidence, 2),
            "confidence_is_calibrated": False,
            "model": "Hybrid CNN + Quantum VQC"
        }
    
    return {
        "prediction": class_labels[class_id],
        "class_id": class_id,
        "confidence": round(confidence, 2),
        "confidence_is_calibrated": False,
        "model": "Hybrid CNN + Quantum VQC"
    }

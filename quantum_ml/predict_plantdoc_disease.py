import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from quantum_ml.cnn_features import CNNFeatureExtractor
from quantum_ml.predict_quantum import _sha256
from quantum_ml.preprocessing import FeatureReducer
from quantum_ml.vqc_model import QuantumVQCModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "plantdoc_disease_v2"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
CLASS_NAMES = ("Disease", "Unspecified Leaf")
MODEL_METADATA_VERSION = 1

try:
    CONFIDENCE_THRESHOLD = float(os.getenv("PREDICTION_CONFIDENCE_THRESHOLD", "0.55"))
except ValueError:
    CONFIDENCE_THRESHOLD = None

_cnn = None
_reducer = None
_vqc = None


def disease_model_readiness():
    if CONFIDENCE_THRESHOLD is None or not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
        return False, "PREDICTION_CONFIDENCE_THRESHOLD must be between 0 and 1"
    if not METADATA_PATH.is_file():
        return False, "PlantDoc disease model has not been trained"
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return False, "PlantDoc disease model metadata is missing or invalid"
        training_counts = metadata.get("training_counts", {})
        artifacts = metadata.get("artifacts", {})
        counts_valid = (
            isinstance(training_counts, dict)
            and all(
                type(training_counts.get(name)) is int and training_counts[name] > 0
                for name in CLASS_NAMES
            )
            and type(metadata.get("training_samples")) is int
            and metadata["training_samples"] == sum(training_counts.values())
        )
        artifact_names = ("scaler.pkl", "pca.pkl", "vqc.model")
        artifacts_valid = (
            isinstance(artifacts, dict)
            and all(
                (MODEL_DIR / name).is_file()
                and artifacts.get(name) == _sha256(MODEL_DIR / name)
                for name in artifact_names
            )
        )
    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
        return False, "PlantDoc disease model metadata is missing or invalid"
    if (
        not isinstance(metadata, dict)
        or metadata.get("model_metadata_version") != MODEL_METADATA_VERSION
        or metadata.get("class_names") != list(CLASS_NAMES)
        or not counts_valid
        or not artifacts_valid
    ):
        return False, "PlantDoc disease model metadata or artifacts do not match the required schema"
    if metadata.get("inference_approved") is not True:
        return False, "PlantDoc disease model is research-only and not approved for live inference"
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


def predict_plantdoc_disease(image: Image.Image):
    ready, message = disease_model_readiness()
    if not ready:
        raise RuntimeError(message)

    cnn, reducer, vqc = _get_models()
    features = cnn.extract_features(image)
    reduced = reducer.reduce(features)
    probabilities = np.asarray(vqc.vqc.predict_proba(reduced), dtype=float)
    if probabilities.shape != (1, len(CLASS_NAMES)):
        raise RuntimeError("PlantDoc disease model did not return both configured class scores")
    probabilities = probabilities[0]
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.isclose(probabilities.sum(), 1.0, atol=1e-3)
    ):
        raise RuntimeError("PlantDoc disease model returned an invalid score distribution")
    class_id = int(probabilities.argmax())
    confidence = float(probabilities[class_id])
    disease_match = class_id == 0 and confidence >= CONFIDENCE_THRESHOLD
    return {
        "prediction": "Disease" if disease_match else "UNCERTAIN",
        "class_id": class_id,
        "matched_class": CLASS_NAMES[class_id],
        "confidence": round(confidence, 2),
        "confidence_is_calibrated": False,
        "classification_scope": "disease_only",
        "model": "PlantDoc disease-match CNN + Qiskit VQC",
    }

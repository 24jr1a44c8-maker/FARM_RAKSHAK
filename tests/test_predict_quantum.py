import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from quantum_ml.predict_quantum import predict_crop


class QuantumPredictionTests(unittest.TestCase):
    def _models(self, scores):
        cnn = Mock()
        cnn.extract_features.return_value = np.zeros(1280)
        reducer = Mock()
        reducer.reduce.return_value = np.zeros((1, 4))
        vqc = SimpleNamespace(vqc=Mock())
        vqc.vqc.predict_proba.return_value = scores
        return cnn, reducer, vqc

    def test_returns_real_model_score_and_uncalibrated_marker(self):
        models = self._models([[0.1, 0.7, 0.1, 0.1]])
        with (
            patch("quantum_ml.predict_quantum.model_readiness", return_value=(True, None)),
            patch("quantum_ml.predict_quantum._get_models", return_value=models),
        ):
            result = predict_crop(Image.new("RGB", (32, 32)))

        self.assertEqual(result["prediction"], "Pest")
        self.assertEqual(result["confidence"], 0.7)
        self.assertFalse(result["confidence_is_calibrated"])

    def test_rejects_incompatible_or_invalid_quantum_scores(self):
        for scores in ([[0.2, 0.8]], [[0.1, 0.1, 0.1, 0.1]], [[0.0, 0.0, 0.0, 0.0]]):
            with self.subTest(scores=scores):
                models = self._models(scores)
                with (
                    patch("quantum_ml.predict_quantum.model_readiness", return_value=(True, None)),
                    patch("quantum_ml.predict_quantum._get_models", return_value=models),
                ):
                    with self.assertRaises(RuntimeError):
                        predict_crop(Image.new("RGB", (32, 32)))


if __name__ == "__main__":
    unittest.main()

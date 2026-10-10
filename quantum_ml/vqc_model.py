import os
import numpy as np
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from quantum_ml.quantum_encoder import get_quantum_encoder

class QuantumVQCModel:
    def __init__(self, num_features=4, model_path='models/quantum_model/vqc.model'):
        self.num_features = num_features
        self.model_path = model_path
        self.vqc = None
        self.labels = {0: "Healthy", 1: "Pest", 2: "Disease", 3: "Non-Crop"}
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            self.vqc = VQC.from_dill(self.model_path)
            
    def build_and_train(self, X_train, y_train):
        feature_map = get_quantum_encoder(self.num_features)
        
        # Trainable parameterized rotation gates and entanglement
        ansatz = RealAmplitudes(num_qubits=self.num_features, reps=2)
        optimizer = COBYLA(maxiter=30)
        sampler = StatevectorSampler()

        self.vqc = VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            loss="cross_entropy"
        )
        
        print("Training Variational Quantum Classifier (VQC)...")
        self.vqc.fit(X_train, y_train)
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.vqc.save(self.model_path)
        print(f"VQC Model saved to {self.model_path}")

    def predict(self, features: np.ndarray):
        if self.vqc is None:
            raise Exception("Model not trained yet.")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        pred = self.vqc.predict(features)
        
        class_idx = np.argmax(pred[0])
        
        return {
            "prediction": self.labels[class_idx],
            "class_id": int(class_idx)
        }

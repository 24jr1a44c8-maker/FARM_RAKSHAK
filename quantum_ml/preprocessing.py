import pickle
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class FeatureReducer:
    def __init__(self, n_components=4, model_dir='models/quantum_model'):
        self.n_components = n_components
        self.model_dir = model_dir
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        self.pca_path = os.path.join(self.model_dir, 'pca.pkl')
        
    def fit(self, features: np.ndarray):
        os.makedirs(self.model_dir, exist_ok=True)
        scaled = self.scaler.fit_transform(features)
        self.pca.fit(scaled)
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(self.pca_path, 'wb') as f:
            pickle.dump(self.pca, f)

    def reduce(self, features: np.ndarray) -> np.ndarray:
        if not hasattr(self.scaler, 'n_features_in_') and os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(self.pca_path, 'rb') as f:
                self.pca = pickle.load(f)

        if not hasattr(self.scaler, 'n_features_in_') or not hasattr(self.pca, 'n_components_'):
            raise RuntimeError("Feature scaler and PCA are not trained; run quantum_ml/train_quantum.py first")

        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        scaled = self.scaler.transform(features)
        return self.pca.transform(scaled)

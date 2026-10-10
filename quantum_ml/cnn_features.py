import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class CNNFeatureExtractor:
    def __init__(self):
        # Using a pretrained MobileNetV2 as the feature extractor
        # weights=models.MobileNet_V2_Weights.DEFAULT is the modern way in torchvision, but we use pretrained=True for backwards compat if needed, or just standard weights.
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Remove the classification head to output raw features (1280 dimensions)
        self.model.classifier = torch.nn.Identity()
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image: Image.Image) -> np.ndarray:
        image_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self.model(image_tensor)
        return features.squeeze().numpy()

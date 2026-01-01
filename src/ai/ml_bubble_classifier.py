"""ML-based bubble classifier using trained CNN."""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class SimpleBubbleCNN(nn.Module):
    """Simple CNN for bubble classification."""

    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class MLBubbleClassifier:
    """ML-based bubble classifier."""

    def __init__(self, model_path='models/bubble_classifier_ml.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleBubbleCNN().to(self.device)

        # Load trained weights
        model_path = Path(model_path)
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded ML bubble classifier from: {model_path}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    def predict(self, bubble_image):
        """Predict if a single bubble is filled.

        Args:
            bubble_image: numpy array (H, W) grayscale bubble image

        Returns:
            (is_filled, confidence): boolean and float
        """
        # Normalize
        img = bubble_image.astype(np.float32) / 255.0

        # Convert to tensor (1, 1, H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)

            # Class 0 = empty, Class 1 = filled
            filled_prob = probabilities[0, 1].item()
            is_filled = filled_prob > 0.5

        return is_filled, filled_prob

    def predict_batch(self, bubble_images):
        """Predict for multiple bubbles.

        Args:
            bubble_images: list of numpy arrays (H, W)

        Returns:
            list of (is_filled, confidence) tuples
        """
        if not bubble_images:
            return []

        # Normalize and stack
        imgs = [(img.astype(np.float32) / 255.0) for img in bubble_images]
        imgs_tensor = torch.stack([torch.from_numpy(img).unsqueeze(0) for img in imgs]).to(self.device)

        with torch.no_grad():
            outputs = self.model(imgs_tensor)
            probabilities = torch.softmax(outputs, dim=1)

            filled_probs = probabilities[:, 1].cpu().numpy()
            results = [(prob > 0.5, float(prob)) for prob in filled_probs]

        return results

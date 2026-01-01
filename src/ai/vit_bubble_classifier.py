"""
Vision Transformer (ViT) Bubble Classifier for OMR Detection

This module provides a ViT-based classifier for detecting filled bubbles
in OMR sheets. It uses a pre-trained ViT-Tiny model and fine-tunes it
for bubble classification.
"""

import os
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. ViT classifier will be disabled.")


class ViTBubbleClassifier(nn.Module):
    """
    Vision Transformer based bubble classifier.

    Uses ViT-Tiny as backbone with custom classification heads:
    - is_filled: Binary classification (filled/unfilled)
    - confidence: Regression (0-1)
    - fill_quality: 4-class (none, light, medium, heavy)
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.1):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise RuntimeError("timm library is required for ViT classifier")

        # Load ViT-Tiny backbone
        self.backbone = timm.create_model(
            'vit_tiny_patch16_224',
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )

        # Get feature dimension
        self.feature_dim = self.backbone.num_features  # 192 for vit_tiny

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Classification heads
        self.is_filled_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.fill_quality_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4)  # none, light, medium, heavy
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 3, 224, 224)

        Returns:
            Dictionary with:
                - is_filled: (batch, 1) logits
                - confidence: (batch, 1) probability
                - fill_quality: (batch, 4) logits
        """
        # Extract features from backbone
        features = self.backbone(x)  # (batch, 192)
        features = self.dropout(features)

        # Classification heads
        is_filled = self.is_filled_head(features)
        confidence = self.confidence_head(features)
        fill_quality = self.fill_quality_head(features)

        return {
            'is_filled': is_filled,
            'confidence': confidence,
            'fill_quality': fill_quality
        }

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict with probabilities.

        Args:
            x: Input tensor

        Returns:
            Dictionary with predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)

            is_filled_prob = torch.sigmoid(outputs['is_filled'])
            is_filled_pred = (is_filled_prob > 0.5).long()

            fill_quality_prob = F.softmax(outputs['fill_quality'], dim=1)
            fill_quality_pred = torch.argmax(fill_quality_prob, dim=1)

            return {
                'is_filled': is_filled_pred.squeeze(),
                'is_filled_prob': is_filled_prob.squeeze(),
                'confidence': outputs['confidence'].squeeze(),
                'fill_quality': fill_quality_pred,
                'fill_quality_prob': fill_quality_prob
            }


class ViTBubbleClassifierInference:
    """
    Wrapper for inference with the ViT bubble classifier.
    """

    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the classifier.

        Args:
            model_path: Path to the trained model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not TIMM_AVAILABLE:
            raise RuntimeError("timm library is required for ViT classifier")

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.model = ViTBubbleClassifier(pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a bubble image for inference.

        Args:
            image: Grayscale or RGB image (H, W) or (H, W, 3)

        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        import cv2

        # Ensure RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize to 224x224
        image = cv2.resize(image, (224, 224))

        # Convert to tensor and normalize
        tensor = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        # Normalize with ImageNet stats
        tensor = (tensor - self.mean) / self.std

        return tensor

    def predict_single(self, image: np.ndarray) -> Dict:
        """
        Predict for a single bubble image.

        Args:
            image: Bubble image (grayscale or RGB)

        Returns:
            Dictionary with prediction results
        """
        tensor = self.preprocess(image)

        with torch.no_grad():
            result = self.model.predict(tensor)

        return {
            'is_filled': bool(result['is_filled'].item()),
            'is_filled_prob': float(result['is_filled_prob'].item()),
            'confidence': float(result['confidence'].item()),
            'fill_quality': int(result['fill_quality'].item()),
            'fill_quality_labels': ['none', 'light', 'medium', 'heavy']
        }

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Predict for a batch of bubble images.

        Args:
            images: List of bubble images

        Returns:
            List of prediction dictionaries
        """
        if not images:
            return []

        # Preprocess all images
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)

        with torch.no_grad():
            result = self.model.predict(batch)

        # Convert to list of dicts
        predictions = []
        for i in range(len(images)):
            pred = {
                'is_filled': bool(result['is_filled'][i].item()),
                'is_filled_prob': float(result['is_filled_prob'][i].item()),
                'confidence': float(result['confidence'][i].item()),
                'fill_quality': int(result['fill_quality'][i].item()),
                'fill_quality_labels': ['none', 'light', 'medium', 'heavy']
            }
            predictions.append(pred)

        return predictions


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            inputs: Logits (batch, 1)
            targets: Binary labels (batch,)

        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs.squeeze(), targets.float(), reduction='none')
        probs = torch.sigmoid(inputs.squeeze())
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce_loss
        return focal_loss.mean()


def load_model(model_path: str = None) -> Optional[ViTBubbleClassifierInference]:
    """
    Load a trained ViT bubble classifier.

    Args:
        model_path: Path to model checkpoint. If None, uses default path.

    Returns:
        ViTBubbleClassifierInference instance or None if not available
    """
    if model_path is None:
        model_path = Path(__file__).parent.parent.parent / 'models' / 'vit_omr' / 'vit_bubble_classifier.pt'

    if not Path(model_path).exists():
        return None

    try:
        return ViTBubbleClassifierInference(str(model_path))
    except Exception as e:
        print(f"Error loading ViT model: {e}")
        return None


# For backwards compatibility
if __name__ == '__main__':
    # Test model creation
    if TIMM_AVAILABLE:
        print("Creating ViT bubble classifier...")
        model = ViTBubbleClassifier(pretrained=True)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        print(f"Output shapes:")
        for k, v in output.items():
            print(f"  {k}: {v.shape}")
    else:
        print("timm not available - cannot create model")

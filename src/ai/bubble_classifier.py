"""
ML-based bubble classifier for OMR detection.

Uses a simple CNN to classify bubbles as filled or empty,
which is more robust than threshold-based detection for
low-contrast scans.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
import json


class BubbleClassifier(nn.Module):
    """Simple CNN for bubble classification (filled vs empty)."""

    def __init__(self):
        super().__init__()

        # Input: 28x28 grayscale bubble image
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BubbleDataset(Dataset):
    """Dataset for bubble classification training."""

    def __init__(self, bubbles: List[np.ndarray], labels: List[int]):
        """
        Args:
            bubbles: List of 28x28 grayscale bubble images
            labels: List of labels (1=filled, 0=empty)
        """
        self.bubbles = bubbles
        self.labels = labels

    def __len__(self):
        return len(self.bubbles)

    def __getitem__(self, idx):
        bubble = self.bubbles[idx].astype(np.float32) / 255.0
        # Add channel dimension
        bubble = bubble[np.newaxis, :, :]
        label = float(self.labels[idx])
        return torch.from_numpy(bubble), torch.tensor(label)


class BubbleClassifierTrainer:
    """Train and use the bubble classifier."""

    # Grid configuration (same as YOLOOMRProcessor)
    TARGET_WIDTH = 2480
    TARGET_HEIGHT = 3507

    NAME_CONFIG = {
        'origin': (193, 690),
        'bubble_size': (28, 28),
        'rows_gap': 51,
        'cols_gap': 54,
        'rows': 26,
        'cols': 25,
    }

    def __init__(self, model_path: Optional[str] = None):
        self.model = BubbleClassifier()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def extract_bubbles_from_image(
        self,
        image_path: str,
        num_cols: int = 25
    ) -> List[Tuple[np.ndarray, int, int]]:
        """Extract all bubbles from an image.

        Returns: List of (bubble_image, col, row) tuples
        """
        image = cv2.imread(image_path)
        if image is None:
            return []

        # Resize to target dimensions
        aligned = cv2.resize(image, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        config = self.NAME_CONFIG
        origin = config['origin']
        bw, bh = config['bubble_size']
        rows_gap = config['rows_gap']
        cols_gap = config['cols_gap']
        num_rows = config['rows']

        bubbles = []
        for col in range(num_cols):
            x = origin[0] + col * cols_gap
            for row in range(num_rows):
                y = origin[1] + row * rows_gap

                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble = gray[y:y+bh, x:x+bw].copy()
                    # Resize to exactly 28x28 if needed
                    if bubble.shape != (28, 28):
                        bubble = cv2.resize(bubble, (28, 28))
                    bubbles.append((bubble, col, row))

        return bubbles

    def generate_training_data(
        self,
        image_dir: str,
        output_dir: str,
        num_images: int = 50
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Generate training data using threshold-based pseudo-labels.

        Uses high-confidence detections from current threshold method
        to bootstrap training data for ML model.
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_bubbles = []
        all_labels = []

        image_paths = sorted(image_dir.glob('*.jpg'))[:num_images]

        for img_path in image_paths:
            bubbles = self.extract_bubbles_from_image(str(img_path))

            # Calculate column-wise statistics for adaptive thresholding
            col_bubbles = {}
            for bubble, col, row in bubbles:
                if col not in col_bubbles:
                    col_bubbles[col] = []
                col_bubbles[col].append((bubble, row, np.mean(bubble)))

            for col, col_data in col_bubbles.items():
                if not col_data:
                    continue

                # Get all darkness values in this column
                darkness_vals = [d[2] for d in col_data]
                min_dark = min(darkness_vals)
                median_dark = np.median(darkness_vals)

                for bubble, row, darkness in col_data:
                    # Adaptive labeling based on column statistics
                    # Filled: significantly darker than column median
                    # Empty: close to or lighter than median

                    if darkness < median_dark * 0.80:  # At least 20% darker than median
                        all_bubbles.append(bubble)
                        all_labels.append(1)  # Filled
                    elif darkness > median_dark * 0.95:  # Within 5% of median = empty
                        all_bubbles.append(bubble)
                        all_labels.append(0)  # Empty
                    # Skip ambiguous cases (between 80% and 95% of median)

        print(f"Generated {len(all_bubbles)} training samples")
        print(f"  Filled: {sum(all_labels)}")
        print(f"  Empty: {len(all_labels) - sum(all_labels)}")

        # Save some examples
        filled_dir = output_dir / 'filled'
        empty_dir = output_dir / 'empty'
        filled_dir.mkdir(exist_ok=True)
        empty_dir.mkdir(exist_ok=True)

        filled_count = 0
        empty_count = 0
        for bubble, label in zip(all_bubbles[:500], all_labels[:500]):
            if label == 1 and filled_count < 100:
                cv2.imwrite(str(filled_dir / f'{filled_count:04d}.png'), bubble)
                filled_count += 1
            elif label == 0 and empty_count < 100:
                cv2.imwrite(str(empty_dir / f'{empty_count:04d}.png'), bubble)
                empty_count += 1

        return all_bubbles, all_labels

    def train(
        self,
        bubbles: List[np.ndarray],
        labels: List[int],
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 0.001
    ):
        """Train the bubble classifier."""
        # Split into train/val
        n = len(bubbles)
        indices = np.random.permutation(n)
        train_size = int(0.8 * n)

        train_bubbles = [bubbles[i] for i in indices[:train_size]]
        train_labels = [labels[i] for i in indices[:train_size]]
        val_bubbles = [bubbles[i] for i in indices[train_size:]]
        val_labels = [labels[i] for i in indices[train_size:]]

        train_dataset = BubbleDataset(train_bubbles, train_labels)
        val_dataset = BubbleDataset(val_bubbles, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        best_val_acc = 0
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_bubbles, batch_labels in train_loader:
                batch_bubbles = batch_bubbles.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_bubbles).squeeze()
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_bubbles, batch_labels in val_loader:
                    batch_bubbles = batch_bubbles.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    outputs = self.model(batch_bubbles).squeeze()
                    predicted = (outputs > 0.5).float()
                    val_correct += (predicted == batch_labels).sum().item()
                    val_total += batch_labels.size(0)

            val_acc = val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss={train_loss/len(train_loader):.4f}, "
                      f"Val Acc={val_acc:.4f}")

        print(f"Best validation accuracy: {best_val_acc:.4f}")

    def predict(self, bubble: np.ndarray) -> Tuple[bool, float]:
        """Predict if a bubble is filled.

        Returns: (is_filled, confidence)
        """
        self.model.eval()

        # Preprocess
        bubble = bubble.astype(np.float32) / 255.0
        bubble = bubble[np.newaxis, np.newaxis, :, :]  # Add batch and channel dims
        bubble_tensor = torch.from_numpy(bubble).to(self.device)

        with torch.no_grad():
            output = self.model(bubble_tensor).item()

        is_filled = output > 0.5
        confidence = output if is_filled else 1 - output

        return is_filled, confidence

    def predict_batch(self, bubbles: List[np.ndarray]) -> List[Tuple[bool, float]]:
        """Predict for multiple bubbles efficiently."""
        if not bubbles:
            return []

        self.model.eval()

        # Stack all bubbles
        batch = np.stack([b.astype(np.float32) / 255.0 for b in bubbles])
        batch = batch[:, np.newaxis, :, :]  # Add channel dim
        batch_tensor = torch.from_numpy(batch).to(self.device)

        with torch.no_grad():
            outputs = self.model(batch_tensor).squeeze().cpu().numpy()

        # Handle single prediction case
        if outputs.ndim == 0:
            outputs = np.array([outputs])

        results = []
        for output in outputs:
            is_filled = output > 0.5
            confidence = output if is_filled else 1 - output
            results.append((is_filled, confidence))

        return results

    def save_model(self, path: str):
        """Save model weights in checkpoint format."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights (handles both checkpoint and direct state_dict formats)."""
        checkpoint = torch.load(path, map_location=self.device)
        # Handle both formats: checkpoint dict or direct state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"Model loaded from {path}")


def train_bubble_classifier():
    """Train the bubble classifier on OMR images."""
    trainer = BubbleClassifierTrainer()

    # Use cross-platform paths relative to this file
    project_dir = Path(__file__).parent.parent.parent  # Go up from src/ai to project root
    image_dir = str(project_dir / 'filledomrsheet')
    output_dir = str(project_dir / 'bubble_training_data')
    model_path = project_dir / 'models' / 'bubble_classifier.pt'

    print("Generating training data...")
    bubbles, labels = trainer.generate_training_data(
        image_dir=image_dir,
        output_dir=output_dir,
        num_images=100
    )

    print("\nTraining model...")
    trainer.train(bubbles, labels, epochs=30)

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_path))

    return trainer


def train_expanded_classifier(max_samples: int = 500000):
    """
    Train bubble classifier on the expanded 5M+ bubble dataset.

    Uses class balancing to handle the 97% empty / 3% filled imbalance.

    Args:
        max_samples: Maximum samples per class for balanced training
    """
    import random

    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / 'bubble_training_data_v2'
    model_path = project_dir / 'models' / 'bubble_classifier_v2.pt'

    print("Loading pre-extracted bubble training data...")
    bubbles = np.load(data_dir / 'bubbles.npy')
    labels = np.load(data_dir / 'labels.npy')

    print(f"Loaded {len(bubbles):,} bubbles")
    print(f"  Filled: {sum(labels):,} ({100*sum(labels)/len(labels):.1f}%)")
    print(f"  Empty: {len(labels)-sum(labels):,} ({100*(len(labels)-sum(labels))/len(labels):.1f}%)")

    # Class balancing: undersample majority class
    filled_indices = np.where(labels == 1)[0]
    empty_indices = np.where(labels == 0)[0]

    # Take all filled samples (minority class) up to max_samples
    n_filled = min(len(filled_indices), max_samples)
    selected_filled = np.random.choice(filled_indices, n_filled, replace=False)

    # Match with equal number of empty samples
    n_empty = min(len(empty_indices), n_filled)  # Same as filled for balance
    selected_empty = np.random.choice(empty_indices, n_empty, replace=False)

    # Combine and shuffle
    selected_indices = np.concatenate([selected_filled, selected_empty])
    np.random.shuffle(selected_indices)

    balanced_bubbles = bubbles[selected_indices]
    balanced_labels = labels[selected_indices]

    print(f"\nBalanced training set: {len(balanced_bubbles):,} samples")
    print(f"  Filled: {sum(balanced_labels):,}")
    print(f"  Empty: {len(balanced_labels)-sum(balanced_labels):,}")

    # Initialize trainer with MPS support
    trainer = BubbleClassifierTrainer()

    # Check for MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        trainer.device = torch.device('mps')
        trainer.model.to(trainer.device)
        print(f"\nUsing MPS (Apple Silicon) for training")
    else:
        print(f"\nUsing device: {trainer.device}")

    # Train with larger batch size for faster training
    print("\nTraining on expanded dataset...")
    trainer.train(
        bubbles=list(balanced_bubbles),
        labels=list(balanced_labels),
        epochs=50,
        batch_size=256,
        lr=0.001
    )

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_path))

    print(f"\nModel saved to {model_path}")
    return trainer


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--expanded':
        train_expanded_classifier()
    else:
        train_bubble_classifier()

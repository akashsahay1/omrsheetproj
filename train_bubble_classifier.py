#!/usr/bin/env python3
"""
Train a CNN classifier to distinguish filled vs unfilled name bubbles.

The key insight: threshold-based detection fails because stray marks
are often darker than actual filled bubbles. A trained classifier
can learn the SHAPE and PATTERN of a properly filled bubble.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install: pip install torch torchvision")
    sys.exit(1)


# Ground truth data
GROUND_TRUTH = {
    "41_629.jpg": "REYANSH KUMAR",
    "41_632.jpg": "YUVRAJ KUMAR MANDAL",
    "41_635.jpg": "ANJALI KUMARI",
    "41_637.jpg": "ROSHAN RAJWAR",
    "41_642.jpg": "SIMANDHYA KUMARI",
    "41_643.jpg": "RAJ KUMAR",
    "41_645.jpg": "ADJNSH KUMAR NAHTO",
    "41_646.jpg": "RAM KUMAR RAWANI",
    "41_648.jpg": "AYUSH KUMAR YADAV",
    "41_649.jpg": "SRISHTI PRIYA",
    "41_650.jpg": "SAKSHAM KUMARI",
    "41_667.jpg": "SREPTI SHARMA",
    "41_672.jpg": "MOHIT KUMAR DAS",
    "41_676.jpg": "RONAK KUMAR MANDAL",
    "41_679.jpg": "PRINCE KUMAR MANDAL",
    "41_682.jpg": "PRINCE KUMAR MANDAL",
    "41_686.jpg": "AJIT KUMAR YADAV",
    "41_687.jpg": "SUMIT KUMAR TURI",
    "41_702.jpg": "SAGAR KUMAR MANDAL",
    "41_705.jpg": "ISHWAR KUMAR",
    "41_724.jpg": "ADARSH KUMAR",
    "41_726.jpg": "SRISHTI SEN",
    "41_739.jpg": "SONAKSHI KUMARI",
    "41_742.jpg": "KHUSHBU KUMARI",
    "41_744.jpg": "PIHU KUMARI",
    "41_749.jpg": "ASHISH KUMAR MANDAL",
    "41_752.jpg": "KHUSHOV MARANDI",
    "41_753.jpg": "JAYASH PRAKHAR",
    "41_756.jpg": "UMA SIMHA",
    "41_760.jpg": "RAVINDRA PAL",
    "41_768.jpg": "KAVYA KUMARI",
    "41_770.jpg": "ANUPA KUSHWAHA",
    "41_771.jpg": "SHRISHTI KUMARI",
    "41_783.jpg": "PRINCE KUMAR",
    "41_784.jpg": "ANKIT KUMAR",
    "41_785.jpg": "SHASHI KUMARI",
    "41_793.jpg": "LADALI KHAN",
    "41_795.jpg": "SHRISHTI KUMARI",
    "41_798.jpg": "SARA ARFIN",
    "41_810.jpg": "KAUSHAL YADAV",
    "41_815.jpg": "AKSHAY KUMAR",
    "41_820.jpg": "PRATIK SONI",
    "41_826.jpg": "RENU KUMARI",
    "41_830.jpg": "JIVIKA KUMARI",
    "41_841.jpg": "TRISHA SONI",
    "41_850.jpg": "FAIZAN RAJA",
    "41_860.jpg": "SNEHA KUMARI",
    "41_870.jpg": "SNEHA KUMARI",
    "41_880.jpg": "IMRAN KHAN",
    "41_890.jpg": "VISHAL KUMAR",
}

# Name field configuration (calibrated)
NAME_CONFIG = {
    'origin': (208, 683),
    'bubble_size': (28, 28),
    'rows_gap': 34,
    'cols_gap': 36,
    'rows': 26,
    'cols': 25,
    'values': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
}


class NameBubbleCNN(nn.Module):
    """CNN for name bubble classification (filled vs unfilled)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),  # Binary: filled vs unfilled
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class BubbleDataset(Dataset):
    """Dataset of bubble images with filled/unfilled labels."""

    def __init__(self, samples: List[Tuple[np.ndarray, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]

        # Resize to 28x28 if needed
        if image.shape[0] != 28 or image.shape[1] != 28:
            image = cv2.resize(image, (28, 28))

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Add channel dimension
        image = np.expand_dims(image, axis=0)

        # Convert to tensor
        image = torch.from_numpy(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def extract_training_data(image_dir: str) -> Tuple[List, List]:
    """Extract labeled bubble samples from images."""
    filled_samples = []
    unfilled_samples = []

    config = NAME_CONFIG
    origin = config['origin']
    bubble_size = config['bubble_size']
    rows_gap = config['rows_gap']
    cols_gap = config['cols_gap']
    num_rows = config['rows']
    num_cols = config['cols']
    values = config['values']

    print("Extracting training data from ground truth images...")

    for filename, ground_truth in GROUND_TRUTH.items():
        image_path = os.path.join(image_dir, filename)

        if not os.path.exists(image_path):
            print(f"  Skipping {filename} - not found")
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Parse ground truth into expected filled positions
        gt_name = ground_truth.upper().replace(" ", "")

        # For each column, determine which row should be filled
        for col in range(min(len(gt_name), num_cols)):
            expected_char = gt_name[col]
            expected_row = values.index(expected_char) if expected_char in values else -1

            x = origin[0] + col * cols_gap

            for row in range(num_rows):
                y = origin[1] + row * rows_gap
                bw, bh = bubble_size

                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble_img = gray[y:y+bh, x:x+bw]

                    # Label based on whether this should be filled
                    is_filled = (row == expected_row)

                    if is_filled:
                        filled_samples.append(bubble_img.copy())
                    else:
                        unfilled_samples.append(bubble_img.copy())

        # Also extract from empty columns (after the name)
        for col in range(len(gt_name), min(len(gt_name) + 5, num_cols)):
            x = origin[0] + col * cols_gap

            for row in range(num_rows):
                y = origin[1] + row * rows_gap
                bw, bh = bubble_size

                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble_img = gray[y:y+bh, x:x+bw]
                    unfilled_samples.append(bubble_img.copy())

    print(f"  Filled samples: {len(filled_samples)}")
    print(f"  Unfilled samples: {len(unfilled_samples)}")

    return filled_samples, unfilled_samples


def analyze_samples(filled: List, unfilled: List):
    """Analyze the extracted samples."""
    print("\n" + "=" * 60)
    print("SAMPLE ANALYSIS")
    print("=" * 60)

    filled_darkness = [np.mean(s) for s in filled]
    unfilled_darkness = [np.mean(s) for s in unfilled]

    print(f"\nFilled bubbles (n={len(filled)}):")
    print(f"  Mean darkness: {np.mean(filled_darkness):.1f}")
    print(f"  Std darkness:  {np.std(filled_darkness):.1f}")
    print(f"  Min darkness:  {np.min(filled_darkness):.1f}")
    print(f"  Max darkness:  {np.max(filled_darkness):.1f}")

    print(f"\nUnfilled bubbles (n={len(unfilled)}):")
    print(f"  Mean darkness: {np.mean(unfilled_darkness):.1f}")
    print(f"  Std darkness:  {np.std(unfilled_darkness):.1f}")
    print(f"  Min darkness:  {np.min(unfilled_darkness):.1f}")
    print(f"  Max darkness:  {np.max(unfilled_darkness):.1f}")

    # Check overlap
    filled_threshold = np.percentile(filled_darkness, 90)
    unfilled_dark = [d for d in unfilled_darkness if d < filled_threshold]
    print(f"\nUnfilled bubbles darker than 90th percentile of filled: {len(unfilled_dark)}")
    print(f"  This shows {100*len(unfilled_dark)/len(unfilled):.1f}% of unfilled are 'problematic'")


def train_classifier(filled: List, unfilled: List, output_path: str):
    """Train the bubble classifier."""
    print("\n" + "=" * 60)
    print("TRAINING CLASSIFIER")
    print("=" * 60)

    # Balance the dataset
    min_samples = min(len(filled), len(unfilled))
    print(f"\nBalancing dataset to {min_samples} samples per class")

    random.shuffle(filled)
    random.shuffle(unfilled)

    filled = filled[:min_samples]
    unfilled = unfilled[:min_samples]

    # Create samples with labels
    samples = [(img, 1) for img in filled] + [(img, 0) for img in unfilled]
    random.shuffle(samples)

    # Split into train/val
    split_idx = int(0.8 * len(samples))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    # Create datasets
    train_dataset = BubbleDataset(train_samples)
    val_dataset = BubbleDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create model
    device = torch.device('mps' if torch.backends.mps.is_available() else
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = NameBubbleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    best_val_acc = 0
    epochs = 30

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.1f}%, "
              f"Val Acc={val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  Saved best model (val_acc={val_acc:.1f}%)")

        scheduler.step()

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to: {output_path}")

    return model


def save_sample_images(filled: List, unfilled: List, output_dir: str):
    """Save sample images for visual inspection."""
    os.makedirs(output_dir, exist_ok=True)

    # Save some filled samples
    for i, img in enumerate(filled[:20]):
        cv2.imwrite(os.path.join(output_dir, f"filled_{i:03d}.png"), img)

    # Save some unfilled samples
    for i, img in enumerate(unfilled[:20]):
        cv2.imwrite(os.path.join(output_dir, f"unfilled_{i:03d}.png"), img)

    # Also save the darkest unfilled samples (potential problems)
    unfilled_with_darkness = [(img, np.mean(img)) for img in unfilled]
    darkest = sorted(unfilled_with_darkness, key=lambda x: x[1])[:20]

    for i, (img, darkness) in enumerate(darkest):
        cv2.imwrite(os.path.join(output_dir, f"unfilled_dark_{i:03d}_{int(darkness)}.png"), img)

    print(f"\nSample images saved to: {output_dir}")


def main():
    image_dir = "/Volumes/Projects/Projects/omrsheetproj/filledomrsheet"
    model_output = "/Volumes/Projects/Projects/omrsheetproj/models/name_bubble_classifier.pt"
    sample_dir = "/Volumes/Projects/Projects/omrsheetproj/bubble_samples"

    # Extract training data
    filled, unfilled = extract_training_data(image_dir)

    if len(filled) < 100 or len(unfilled) < 100:
        print("Not enough training data!")
        return

    # Analyze samples
    analyze_samples(filled, unfilled)

    # Save sample images for inspection
    save_sample_images(filled, unfilled, sample_dir)

    # Train classifier
    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    model = train_classifier(filled, unfilled, model_output)


if __name__ == "__main__":
    main()

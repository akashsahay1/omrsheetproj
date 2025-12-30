"""
Extract bubbles from all training images for classifier retraining.

Processes:
- /Volumes/Projects/traindata/Chayan/ (7,671 images)
- /Volumes/Projects/traindata/Khoj/ (8,360 images)
- filledomrsheet/ (302 images)

Usage:
    python scripts/extract_training_bubbles.py
"""

import cv2
import numpy as np
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import random


class BubbleExtractor:
    """Extract bubble images from OMR sheets for training."""

    TARGET_WIDTH = 2480
    TARGET_HEIGHT = 3507

    # Grid configs for different batches (slightly different origins)
    NAME_CONFIG_KHOJ = {
        'origin': (189, 671),
        'bubble_size': (33, 33),
        'rows_gap': 52,
        'cols_gap': 54,
        'rows': 26,
        'cols': 25,
    }

    NAME_CONFIG_CHAYAN = {
        'origin': (180, 710),
        'bubble_size': (33, 33),
        'rows_gap': 52,
        'cols_gap': 54,
        'rows': 26,
        'cols': 25,
    }

    # Answer section config (same for both batches)
    ANSWER_CONFIG = {
        'origin': (189, 2057),
        'bubble_size': (33, 33),
        'rows_gap': 52,
        'cols_gap': 54,
        'rows': 4,  # A, B, C, D
        'cols': 60,  # 60 questions
    }

    def __init__(self, output_dir: str = 'bubble_training_data_v2'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def extract_from_all_sources(self) -> Tuple[int, int]:
        """Extract bubbles from all training sources."""
        all_bubbles = []
        all_labels = []
        metadata = []

        # Source 1: Khoj batch
        khoj_dir = Path('/Volumes/Projects/traindata/Khoj')
        if khoj_dir.exists():
            print(f"\nProcessing Khoj batch from {khoj_dir}")
            b, l, m = self._extract_from_directory(khoj_dir, 'khoj')
            all_bubbles.extend(b)
            all_labels.extend(l)
            metadata.extend(m)
            print(f"  Extracted {len(b)} bubbles (Filled: {sum(l)}, Empty: {len(l)-sum(l)})")

        # Source 2: Chayan batch
        chayan_dir = Path('/Volumes/Projects/traindata/Chayan')
        if chayan_dir.exists():
            print(f"\nProcessing Chayan batch from {chayan_dir}")
            b, l, m = self._extract_from_directory(chayan_dir, 'chayan')
            all_bubbles.extend(b)
            all_labels.extend(l)
            metadata.extend(m)
            print(f"  Extracted {len(b)} bubbles (Filled: {sum(l)}, Empty: {len(l)-sum(l)})")

        # Source 3: Local filledomrsheet
        local_dir = Path(__file__).parent.parent / 'filledomrsheet'
        if local_dir.exists():
            print(f"\nProcessing local batch from {local_dir}")
            b, l, m = self._extract_from_directory(local_dir, 'khoj')  # Same format as Khoj
            all_bubbles.extend(b)
            all_labels.extend(l)
            metadata.extend(m)
            print(f"  Extracted {len(b)} bubbles (Filled: {sum(l)}, Empty: {len(l)-sum(l)})")

        # Save metadata
        print(f"\nTotal: {len(all_bubbles)} bubbles")
        print(f"  Filled: {sum(all_labels)}")
        print(f"  Empty: {len(all_labels) - sum(all_labels)}")

        # Save training data
        self._save_training_data(all_bubbles, all_labels, metadata)

        return len(all_bubbles), sum(all_labels)

    def _extract_from_directory(
        self,
        image_dir: Path,
        batch_type: str = 'khoj',
        max_images: Optional[int] = None
    ) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Extract all bubbles from a directory with pseudo-labels."""
        config = self.NAME_CONFIG_CHAYAN if batch_type == 'chayan' else self.NAME_CONFIG_KHOJ

        all_bubbles = []
        all_labels = []
        all_metadata = []

        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.JPG'))

        if max_images:
            image_paths = image_paths[:max_images]

        for img_path in tqdm(image_paths, desc=f"Extracting from {batch_type}"):
            bubbles, labels, meta = self._extract_from_image(str(img_path), config, batch_type)
            all_bubbles.extend(bubbles)
            all_labels.extend(labels)
            all_metadata.extend(meta)

        return all_bubbles, all_labels, all_metadata

    def _extract_from_image(
        self,
        image_path: str,
        config: Dict,
        batch_type: str
    ) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """Extract bubbles from single image with adaptive pseudo-labeling."""
        image = cv2.imread(image_path)
        if image is None:
            return [], [], []

        # Resize to standard dimensions
        aligned = cv2.resize(image, (self.TARGET_WIDTH, self.TARGET_HEIGHT))
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

        bubbles = []
        labels = []
        metadata = []

        origin = config['origin']
        bw, bh = config['bubble_size']
        filename = Path(image_path).stem

        # Extract bubbles column by column for name field
        for col in range(config['cols']):
            x = origin[0] + col * config['cols_gap']
            col_bubbles = []

            for row in range(config['rows']):
                y = origin[1] + row * config['rows_gap']

                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble = gray[y:y+bh, x:x+bw].copy()
                    bubble_resized = cv2.resize(bubble, (28, 28))
                    darkness = np.mean(bubble)
                    col_bubbles.append((bubble_resized, darkness, row, col))

            if not col_bubbles:
                continue

            # Adaptive labeling using column statistics
            darkness_vals = [d[1] for d in col_bubbles]
            median_dark = np.median(darkness_vals)
            min_dark = min(darkness_vals)
            max_dark = max(darkness_vals)
            gap = max_dark - min_dark

            # Only label if there's a clear distinction
            if gap < 20:
                # Low contrast column - skip to avoid noisy labels
                continue

            for bubble, darkness, row, col in col_bubbles:
                # High-confidence filled (much darker than median)
                if darkness < median_dark * 0.75 and darkness < 180:
                    bubbles.append(bubble)
                    labels.append(1)
                    metadata.append({
                        'filename': filename,
                        'batch': batch_type,
                        'col': col,
                        'row': row,
                        'darkness': float(darkness),
                        'label': 1
                    })
                # High-confidence empty (close to or above median)
                elif darkness > median_dark * 0.95 and darkness > 200:
                    bubbles.append(bubble)
                    labels.append(0)
                    metadata.append({
                        'filename': filename,
                        'batch': batch_type,
                        'col': col,
                        'row': row,
                        'darkness': float(darkness),
                        'label': 0
                    })
                # Skip ambiguous cases to avoid noisy labels

        return bubbles, labels, metadata

    def _save_training_data(
        self,
        bubbles: List[np.ndarray],
        labels: List[int],
        metadata: List[Dict]
    ):
        """Save extracted training data."""
        # Create numpy arrays
        X = np.array(bubbles)
        y = np.array(labels)

        # Save as numpy files
        np.save(self.output_dir / 'bubbles.npy', X)
        np.save(self.output_dir / 'labels.npy', y)

        # Save metadata
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Also save sample images for verification
        samples_dir = self.output_dir / 'samples'
        samples_dir.mkdir(exist_ok=True)

        # Save 50 random filled and 50 random empty samples
        filled_indices = [i for i, l in enumerate(labels) if l == 1]
        empty_indices = [i for i, l in enumerate(labels) if l == 0]

        random.seed(42)
        for idx in random.sample(filled_indices, min(50, len(filled_indices))):
            cv2.imwrite(str(samples_dir / f'filled_{idx}.png'), bubbles[idx])

        for idx in random.sample(empty_indices, min(50, len(empty_indices))):
            cv2.imwrite(str(samples_dir / f'empty_{idx}.png'), bubbles[idx])

        print(f"\nSaved training data to {self.output_dir}")
        print(f"  bubbles.npy: {X.shape}")
        print(f"  labels.npy: {y.shape}")
        print(f"  metadata.json: {len(metadata)} entries")
        print(f"  samples/: 100 sample images")


def main():
    """Main entry point."""
    extractor = BubbleExtractor()
    total, filled = extractor.extract_from_all_sources()

    print(f"\n{'='*60}")
    print(f"Extraction Complete!")
    print(f"Total bubbles: {total}")
    print(f"Filled: {filled} ({100*filled/total:.1f}%)")
    print(f"Empty: {total-filled} ({100*(total-filled)/total:.1f}%)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

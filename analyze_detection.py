#!/usr/bin/env python3
"""Analyze bubble detection in detail for debugging."""

import sys
sys.path.insert(0, '/Volumes/Projects/Projects/omrsheetproj')

import cv2
import numpy as np
from pathlib import Path

# Constants from YOLOOMRProcessor
TARGET_WIDTH = 2480
TARGET_HEIGHT = 3507

NAME_CONFIG = {
    'origin': (193, 690),
    'bubble_size': (28, 28),
    'rows_gap': 51,
    'cols_gap': 54,
    'rows': 26,
    'cols': 25,
    'values': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
}

def analyze_image(image_path):
    """Analyze bubble detection for a single image."""
    print(f"\nAnalyzing: {image_path}")
    print("=" * 80)

    # Load and resize
    image = cv2.imread(str(image_path))
    if image is None:
        print("Failed to load image")
        return

    aligned = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    config = NAME_CONFIG
    origin = config['origin']
    bubble_size = config['bubble_size']
    rows_gap = config['rows_gap']
    cols_gap = config['cols_gap']
    num_rows = config['rows']
    num_cols = config['cols']
    values = config['values']

    result_chars = []

    # Analyze first 15 columns (typical name length)
    for col in range(15):
        x = origin[0] + col * cols_gap

        bubble_data = []
        for row in range(num_rows):
            y = origin[1] + row * rows_gap
            bw, bh = bubble_size

            if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                bubble = gray[y:y+bh, x:x+bw]
                darkness = np.mean(bubble)
                bubble_data.append((row, values[row], darkness))

        if not bubble_data:
            result_chars.append(' ')
            continue

        # Sort by darkness
        sorted_bubbles = sorted(bubble_data, key=lambda x: x[2])
        darkest_row, darkest_char, darkest_val = sorted_bubbles[0]

        # Calculate statistics
        all_vals = [d[2] for d in bubble_data]
        median_val = np.median(all_vals)
        min_val = min(all_vals)
        max_val = max(all_vals)

        # Current threshold
        threshold = median_val * 0.75

        # Check if significantly darker
        is_filled = darkest_val < threshold

        if is_filled:
            result_chars.append(darkest_char)
            status = "FILLED"
        else:
            result_chars.append('_')
            status = "empty"

        # Show detailed info for this column
        print(f"Col {col:2d}: {status:6s} | Darkest: {darkest_char}={darkest_val:.0f} | "
              f"Median: {median_val:.0f} | Threshold: {threshold:.0f} | "
              f"Range: {min_val:.0f}-{max_val:.0f}")

        # Show top 3 darkest if there's competition
        if len(sorted_bubbles) >= 3:
            top3 = sorted_bubbles[:3]
            diffs = [median_val - d[2] for d in top3]
            if diffs[1] > 20:  # Close competition
                print(f"       Top 3: " + ", ".join(
                    f"{d[1]}={d[2]:.0f}" for d in top3))

    detected = ''.join(result_chars).strip()
    print("-" * 80)
    print(f"Detected name: {detected}")

    return detected

def main():
    test_dir = Path('/Volumes/Projects/Projects/omrsheetproj/filledomrsheet')

    # Test on a few specific images with known issues
    problem_images = [
        '41_627.jpg',  # CHANDNNIKUMRRI -> should be CHANDNI KUMARI
        '41_633.jpg',  # NJLKUMR -> should be NIKHIL KUMAR
        '41_625.jpg',  # Working: SHIVANSH KUMAR
    ]

    for img_name in problem_images:
        img_path = test_dir / img_name
        if img_path.exists():
            analyze_image(img_path)
            print()

if __name__ == "__main__":
    main()

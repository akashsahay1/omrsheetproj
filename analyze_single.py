#!/usr/bin/env python3
"""Analyze a single image in detail."""

import sys
sys.path.insert(0, '/Volumes/Projects/Projects/omrsheetproj')

import cv2
import numpy as np
from pathlib import Path

TARGET_WIDTH = 2480
TARGET_HEIGHT = 3507

NAME_CONFIG = {
    'origin': (193, 690),
    'bubble_size': (28, 28),
    'rows_gap': 51,
    'cols_gap': 54,
}

def analyze(image_path):
    print(f"\nAnalyzing: {image_path}")
    print("=" * 90)

    image = cv2.imread(str(image_path))
    aligned = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    config = NAME_CONFIG
    origin = config['origin']
    bw, bh = config['bubble_size']
    rows_gap = config['rows_gap']
    cols_gap = config['cols_gap']
    values = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    result_chars = []

    for col in range(12):  # First 12 columns
        x = origin[0] + col * cols_gap

        bubble_data = []
        for row in range(26):
            y = origin[1] + row * rows_gap
            if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                bubble = gray[y:y+bh, x:x+bw]
                darkness = np.mean(bubble)
                bubble_data.append((row, values[row], darkness))

        sorted_bubbles = sorted(bubble_data, key=lambda x: x[2])
        darkest_row, darkest_char, darkest_val = sorted_bubbles[0]
        second_row, second_char, second_val = sorted_bubbles[1]

        all_vals = [d[2] for d in bubble_data]
        median_val = np.median(all_vals)
        gap = second_val - darkest_val

        # Current detection logic
        is_filled = (
            darkest_val < median_val * 0.85
            or (gap > 25 and darkest_val < median_val * 0.95)
        )

        if is_filled:
            result_chars.append(darkest_char)
            status = "FILLED"
        else:
            result_chars.append('_')
            status = "empty"

        print(f"Col {col:2d}: {status:6s} | "
              f"Darkest: {darkest_char}={darkest_val:5.1f} | "
              f"2nd: {second_char}={second_val:5.1f} | "
              f"Gap: {gap:5.1f} | "
              f"Median: {median_val:5.1f} | "
              f"Threshold(0.85): {median_val*0.85:5.1f}")

    print("-" * 90)
    print(f"Detected: {''.join(result_chars)}")
    print(f"Expected: KARAN_MAHTO (with space at col 5)")

# Analyze KARAN MAHTO image
analyze('/Volumes/Projects/Projects/omrsheetproj/filledomrsheet/41_640.jpg')

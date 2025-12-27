#!/usr/bin/env python3
"""
Diagnostic script to understand why name detection is failing.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ai.yolo_omr_processor import YOLOOMRProcessor


def diagnose_name_field(image_path: str, expected_name: str):
    """Diagnose name field detection for a single image."""
    processor = YOLOOMRProcessor()

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    config = processor.FIELD_CONFIGS['name']
    origin = config['origin']
    bubble_size = config['bubble_size']
    rows_gap = config['rows_gap']
    cols_gap = config['cols_gap']
    num_rows = config['rows']
    num_cols = config['cols']
    values = config['values']

    print(f"\nDiagnosing: {os.path.basename(image_path)}")
    print(f"Expected name: {expected_name}")
    print(f"Image shape: {image.shape}")
    print(f"\nName field config:")
    print(f"  Origin: {origin}")
    print(f"  Bubble size: {bubble_size}")
    print(f"  Rows gap: {rows_gap}, Cols gap: {cols_gap}")
    print(f"  Grid: {num_rows} rows x {num_cols} cols")

    # Analyze first 10 columns
    print(f"\n{'Col':<4} {'Exp':<4} {'All Values':<60} {'Det':<4} {'Gap':<8} {'Status'}")
    print("-" * 100)

    expected_chars = expected_name.upper().replace(" ", "")
    detected_chars = []

    for col in range(min(15, num_cols)):
        x = origin[0] + col * cols_gap

        # Get expected character for this column
        exp_char = expected_chars[col] if col < len(expected_chars) else " "
        exp_row = values.index(exp_char) if exp_char in values else -1

        # Sample all rows in this column
        row_values = []
        for row in range(num_rows):
            y = origin[1] + row * rows_gap
            bw, bh = bubble_size

            if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                bubble = gray[y:y+bh, x:x+bw]
                darkness = np.mean(bubble)
                row_values.append((row, darkness, values[row]))

        if not row_values:
            continue

        # Sort by darkness
        sorted_rows = sorted(row_values, key=lambda x: x[1])
        darkest = sorted_rows[0]
        second = sorted_rows[1] if len(sorted_rows) > 1 else sorted_rows[0]

        gap = second[1] - darkest[1]
        median_val = np.median([v[1] for v in row_values])
        gap_to_median = median_val - darkest[1]

        # Show top 3 darkest
        top3 = sorted_rows[:3]
        vals_str = ", ".join([f"{v[2]}:{v[1]:.0f}" for v in top3])

        # Detection logic from processor
        is_filled = (
            (gap_to_median > 40 and darkest[1] < 235)
            or (gap_to_median > 20 and darkest[1] < 230)
            or (gap_to_median > 12 and darkest[1] < 220)
            or (gap_to_median > 8 and darkest[1] < 200)
        )

        det_char = darkest[2] if is_filled else "_"
        detected_chars.append(det_char)

        # Check if correct
        status = "OK" if darkest[0] == exp_row and is_filled else ""
        if exp_row != -1 and darkest[0] == exp_row and not is_filled:
            status = "MISSED"
        elif is_filled and darkest[0] != exp_row:
            status = "WRONG"

        print(f"{col:<4} {exp_char:<4} {vals_str:<60} {det_char:<4} {gap:.1f} (m:{gap_to_median:.1f}) {status}")

    detected_name = "".join(detected_chars).replace("_", "")
    print(f"\nDetected: '{detected_name}'")
    print(f"Expected: '{expected_name}'")

    # Visualize the name field region
    viz_image = image.copy()
    x1, y1 = origin
    x2 = origin[0] + num_cols * cols_gap + bubble_size[0]
    y2 = origin[1] + num_rows * rows_gap + bubble_size[1]

    cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw grid
    for col in range(num_cols):
        x = origin[0] + col * cols_gap
        for row in range(num_rows):
            y = origin[1] + row * rows_gap
            bw, bh = bubble_size
            cv2.rectangle(viz_image, (x, y), (x+bw, y+bh), (255, 0, 0), 1)

    output_path = f"diagnose_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, viz_image)
    print(f"\nVisualization saved to: {output_path}")


def main():
    # Test with a few sample images
    test_cases = [
        ("/Volumes/Projects/Projects/omrsheetproj/filledomrsheet/41_629.jpg", "REYANSH KUMAR"),
        ("/Volumes/Projects/Projects/omrsheetproj/filledomrsheet/41_632.jpg", "YUVRAJ KUMAR MANDAL"),
        ("/Volumes/Projects/Projects/omrsheetproj/filledomrsheet/41_815.jpg", "AKSHAY KUMAR"),
    ]

    for image_path, expected_name in test_cases:
        diagnose_name_field(image_path, expected_name)
        print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

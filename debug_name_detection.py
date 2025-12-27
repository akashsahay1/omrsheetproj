#!/usr/bin/env python3
"""
Debug the name bubble detection to understand why it's returning empty.
"""

import cv2
import numpy as np
from pathlib import Path

TARGET_WIDTH = 2480
TARGET_HEIGHT = 3507


def debug_name_detection(image_path: str):
    """Debug the name bubble detection."""
    image = cv2.imread(image_path)
    aligned = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    # Name grid config (calibrated values)
    config = {
        'origin': (189, 671),
        'bubble_size': (33, 33),
        'rows_gap': 52,
        'cols_gap': 54,
        'rows': 26,
        'cols': 25,
        'values': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
    }

    origin = config['origin']
    bubble_size = config['bubble_size']
    rows_gap = config['rows_gap']
    cols_gap = config['cols_gap']
    num_rows = config['rows']
    num_cols = config['cols']
    values = config['values']

    print(f"Config: origin={origin}, bubble_size={bubble_size}")
    print(f"        rows_gap={rows_gap}, cols_gap={cols_gap}")
    print()

    # Sample first few columns
    detected_name = ""

    for col in range(min(15, num_cols)):
        x = origin[0] + col * cols_gap

        # Sample all rows for this column
        row_values = []
        for row in range(num_rows):
            y = origin[1] + row * rows_gap
            bw, bh = bubble_size

            if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                bubble = gray[y:y+bh, x:x+bw]
                darkness = np.mean(bubble)
                row_values.append((row, darkness, values[row]))

        if row_values:
            # Sort by darkness (lowest = darkest = filled)
            sorted_rows = sorted(row_values, key=lambda x: x[1])
            darkest = sorted_rows[0]
            second_darkest = sorted_rows[1] if len(sorted_rows) > 1 else (0, 255, '')
            median_darkness = np.median([v[1] for v in row_values])

            # Calculate gap between darkest and median
            gap = median_darkness - darkest[1]

            print(f"Col {col}: Darkest={darkest[2]}({darkest[1]:.0f}), "
                  f"2nd={second_darkest[2]}({second_darkest[1]:.0f}), "
                  f"Median={median_darkness:.0f}, Gap={gap:.0f}")

            # Detect if filled (darkest is significantly darker than median)
            THRESHOLD = 25  # Gap required to consider filled
            if gap >= THRESHOLD:
                detected_name += darkest[2]
                print(f"        -> Detected: {darkest[2]}")
            else:
                # Check if this might be an empty column (no bubble filled)
                print(f"        -> No clear fill (gap={gap:.0f} < {THRESHOLD})")

    print(f"\nDetected name: '{detected_name}'")

    # Create visualization
    viz = aligned.copy()

    # Draw the grid
    for col in range(min(15, num_cols)):
        x = origin[0] + col * cols_gap
        for row in range(num_rows):
            y = origin[1] + row * rows_gap
            bw, bh = bubble_size
            cv2.rectangle(viz, (x, y), (x+bw, y+bh), (0, 0, 255), 1)

    cv2.imwrite("debug_name_grid.jpg", viz)
    print("\nSaved: debug_name_grid.jpg")

    return detected_name


if __name__ == "__main__":
    # Use cross-platform path relative to this script
    project_dir = Path(__file__).parent
    image_path = project_dir / "filledomrsheet" / "41_629.jpg"
    result = debug_name_detection(str(image_path))
    print(f"\nFinal result: '{result}'")

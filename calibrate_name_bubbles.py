#!/usr/bin/env python3
"""
Calibrate the name bubble grid coordinates by detecting circles.
"""

import cv2
import numpy as np
from pathlib import Path

TARGET_WIDTH = 2480
TARGET_HEIGHT = 3507


def calibrate_name_grid(image_path: str):
    """Detect and calibrate the name bubble grid."""
    image = cv2.imread(image_path)
    aligned = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

    # Focus on the name bubble grid region (below handwritten name)
    # Based on visual inspection: y=650-1350, x=30-950
    roi_y1, roi_y2 = 650, 1350
    roi_x1, roi_x2 = 30, 950

    roi = aligned[roi_y1:roi_y2, roi_x1:roi_x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=25,  # Minimum distance between circle centers
        param1=50,
        param2=20,
        minRadius=10,
        maxRadius=18
    )

    if circles is None:
        print("No circles detected!")
        return None

    circles = np.uint16(np.around(circles[0]))
    print(f"Detected {len(circles)} circles")

    # Convert to absolute coordinates
    circle_list = [(roi_x1 + c[0], roi_y1 + c[1], c[2]) for c in circles]

    # Sort by y then x to organize into rows
    circle_list.sort(key=lambda c: (c[1], c[0]))

    # Group into rows (circles with similar y values)
    rows = []
    current_row = [circle_list[0]]
    current_y = circle_list[0][1]

    for circle in circle_list[1:]:
        if abs(circle[1] - current_y) < 20:
            current_row.append(circle)
        else:
            if len(current_row) >= 10:  # Valid row should have many columns
                rows.append(sorted(current_row, key=lambda c: c[0]))
            current_row = [circle]
            current_y = circle[1]

    if len(current_row) >= 10:
        rows.append(sorted(current_row, key=lambda c: c[0]))

    print(f"\nFound {len(rows)} rows with 10+ circles")

    # Analyze the grid structure
    if len(rows) >= 2:
        # First row analysis
        first_row = rows[0]
        first_circle = first_row[0]

        # Calculate average radius
        all_radii = [c[2] for c in circle_list]
        avg_radius = np.mean(all_radii)

        # Origin (top-left corner of first bubble)
        origin_x = int(first_circle[0] - avg_radius)
        origin_y = int(first_circle[1] - avg_radius)

        # Column gap (horizontal spacing)
        col_gaps = []
        for row in rows[:5]:  # Use first 5 rows
            for i in range(min(15, len(row) - 1)):
                col_gaps.append(row[i+1][0] - row[i][0])
        cols_gap = np.median(col_gaps) if col_gaps else 36

        # Row gap (vertical spacing)
        row_gaps = []
        for i in range(min(10, len(rows) - 1)):
            row_gaps.append(rows[i+1][0][1] - rows[i][0][1])
        rows_gap = np.median(row_gaps) if row_gaps else 34

        # Bubble size (diameter)
        bubble_size = int(avg_radius * 2)

        print(f"\n{'='*60}")
        print("CALIBRATED NAME GRID PARAMETERS")
        print(f"{'='*60}")
        print(f"Origin: ({origin_x}, {origin_y})")
        print(f"Column gap: {cols_gap:.1f}")
        print(f"Row gap: {rows_gap:.1f}")
        print(f"Bubble size: {bubble_size}")
        print(f"Average radius: {avg_radius:.1f}")
        print(f"Number of rows detected: {len(rows)}")
        print(f"Columns in first row: {len(first_row)}")

        # Print first few rows details
        print(f"\nFirst 5 rows (y coordinates):")
        for i, row in enumerate(rows[:5]):
            letter = chr(ord('A') + i)
            print(f"  Row {letter}: y={row[0][1]}, n_cols={len(row)}")

        # Create visualization
        viz = aligned.copy()

        # Draw detected circles in green
        for c in circle_list:
            cv2.circle(viz, (c[0], c[1]), c[2], (0, 255, 0), 1)

        # Draw calculated grid in red
        for col in range(25):
            x = int(origin_x + col * cols_gap)
            for row in range(26):
                y = int(origin_y + row * rows_gap)
                cv2.rectangle(viz, (x, y), (x + bubble_size, y + bubble_size), (0, 0, 255), 1)

        cv2.imwrite("name_grid_calibration.jpg", viz)
        print(f"\nVisualization saved to: name_grid_calibration.jpg")

        return {
            'origin': (origin_x, origin_y),
            'cols_gap': cols_gap,
            'rows_gap': rows_gap,
            'bubble_size': (bubble_size, bubble_size),
            'rows': 26,  # A-Z
            'cols': 25,  # Max name length
        }

    return None


if __name__ == "__main__":
    # Use cross-platform path relative to this script
    project_dir = Path(__file__).parent
    sample_image = project_dir / "filledomrsheet" / "41_629.jpg"
    result = calibrate_name_grid(str(sample_image))

    if result:
        print(f"\nPython config:")
        print(f"'name': {{")
        print(f"    'origin': {result['origin']},")
        print(f"    'bubble_size': {result['bubble_size']},")
        print(f"    'rows_gap': {result['rows_gap']:.0f},")
        print(f"    'cols_gap': {result['cols_gap']:.0f},")
        print(f"    'rows': 26,")
        print(f"    'cols': 25,")
        print(f"    'values': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),")
        print(f"}},")

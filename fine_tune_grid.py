#!/usr/bin/env python3
"""
Fine-tune the grid by detecting actual bubble circles.
"""

import cv2
import numpy as np


def detect_small_circles(image_path: str):
    """Detect small bubble circles."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Focus on name bubble region
    roi_y1, roi_y2 = 680, 1400
    roi_x1, roi_x2 = 160, 1000

    roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

    # Apply bilateral filter to smooth while preserving edges
    filtered = cv2.bilateralFilter(roi, 9, 75, 75)

    # Detect circles with smaller radii
    circles = cv2.HoughCircles(
        filtered,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,  # Closer circles
        param1=50,
        param2=18,   # Lower threshold for more detections
        minRadius=6,
        maxRadius=14
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Found {len(circles[0])} circles")

        # Convert to image coordinates
        circle_list = [(roi_x1 + c[0], roi_y1 + c[1], c[2]) for c in circles[0]]

        # Sort by y then x
        circle_list.sort(key=lambda c: (c[1], c[0]))

        # Group into rows
        rows = []
        current_row = [circle_list[0]]

        for circle in circle_list[1:]:
            if abs(circle[1] - current_row[0][1]) < 20:
                current_row.append(circle)
            else:
                if len(current_row) >= 15:  # Name grid should have many columns
                    rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [circle]

        if len(current_row) >= 15:
            rows.append(sorted(current_row, key=lambda c: c[0]))

        print(f"\nFound {len(rows)} rows with 15+ circles")

        # Analyze each row
        for i, row in enumerate(rows[:26]):
            y_avg = sum(c[1] for c in row) / len(row)
            x_first = row[0][0]
            n_cols = len(row)

            if len(row) > 1:
                col_gaps = [row[j+1][0] - row[j][0] for j in range(min(10, len(row)-1))]
                avg_gap = sum(col_gaps) / len(col_gaps)
            else:
                avg_gap = 0

            print(f"Row {i}: y={y_avg:.0f}, x_first={x_first}, n_cols={n_cols}, avg_gap={avg_gap:.1f}")

        # Calculate parameters
        if len(rows) >= 2:
            first_row = rows[0]
            radius = np.mean([c[2] for c in first_row[:10]])

            origin_x = int(first_row[0][0] - radius)
            origin_y = int(first_row[0][1] - radius)

            col_gaps = []
            for row in rows[:5]:
                if len(row) > 1:
                    for j in range(min(15, len(row)-1)):
                        col_gaps.append(row[j+1][0] - row[j][0])
            cols_gap = sum(col_gaps) / len(col_gaps) if col_gaps else 30

            row_gaps = [rows[i+1][0][1] - rows[i][0][1] for i in range(min(10, len(rows)-1))]
            rows_gap = sum(row_gaps) / len(row_gaps) if row_gaps else 30

            bubble_size = int(radius * 2)

            print(f"\n{'='*60}")
            print("DETECTED PARAMETERS")
            print(f"{'='*60}")
            print(f"Origin: ({origin_x}, {origin_y})")
            print(f"Column gap: {cols_gap:.1f}")
            print(f"Row gap: {rows_gap:.1f}")
            print(f"Bubble size: {bubble_size}")

            # Create visualization
            viz = image.copy()

            # Draw detected circles in green
            for c in circle_list:
                cv2.circle(viz, (c[0], c[1]), c[2], (0, 255, 0), 1)

            # Draw calculated grid in red
            for col in range(25):
                x = int(origin_x + col * cols_gap)
                for row in range(26):
                    y = int(origin_y + row * rows_gap)
                    cv2.rectangle(viz, (x, y), (x + bubble_size, y + bubble_size), (0, 0, 255), 1)

            cv2.imwrite("fine_tune_grid.jpg", viz)
            print("\nSaved: fine_tune_grid.jpg")

            # Also save just the detected circles
            viz2 = image.copy()
            for c in circle_list:
                cv2.circle(viz2, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.imwrite("detected_circles_only.jpg", viz2)
            print("Saved: detected_circles_only.jpg")

            return {
                'origin': (origin_x, origin_y),
                'cols_gap': cols_gap,
                'rows_gap': rows_gap,
                'bubble_size': bubble_size
            }

    return None


def main():
    image_path = "/Volumes/Projects/Projects/omrsheetproj/filledomrsheet/41_629.jpg"
    result = detect_small_circles(image_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Precise calibration by visually examining bubble positions.
"""

import cv2
import numpy as np


def extract_and_analyze_name_region(image_path: str):
    """Extract the name bubble region and analyze it closely."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"Image size: {image.shape}")

    # The name section appears to be between y=680 and y=1380 approximately
    # And between x=150 and x=1050

    # Let's crop just the name bubble grid area
    name_y1, name_y2 = 680, 1420
    name_x1, name_x2 = 150, 1100

    name_region = gray[name_y1:name_y2, name_x1:name_x2]
    name_color = image[name_y1:name_y2, name_x1:name_x2].copy()

    print(f"Name region size: {name_region.shape}")

    # Find all dark circles (potential bubbles)
    # Use adaptive thresholding
    blurred = cv2.GaussianBlur(name_region, (5, 5), 0)

    # Use HoughCircles for better bubble detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=25,
        param1=50,
        param2=20,
        minRadius=8,
        maxRadius=16
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Found {len(circles[0])} circles")

        # Draw circles on the image
        for c in circles[0]:
            cv2.circle(name_color, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(name_color, (c[0], c[1]), 2, (0, 0, 255), -1)

        # Convert to absolute coordinates
        circle_list = [(name_x1 + c[0], name_y1 + c[1], c[2]) for c in circles[0]]

        # Sort by y then x
        circle_list.sort(key=lambda c: (c[1], c[0]))

        # Group into rows
        rows = []
        current_row = [circle_list[0]]
        current_y = circle_list[0][1]

        for circle in circle_list[1:]:
            if abs(circle[1] - current_y) < 20:
                current_row.append(circle)
            else:
                if len(current_row) >= 10:
                    rows.append(sorted(current_row, key=lambda c: c[0]))
                current_row = [circle]
                current_y = circle[1]

        if len(current_row) >= 10:
            rows.append(sorted(current_row, key=lambda c: c[0]))

        print(f"\nFound {len(rows)} rows with 10+ circles")

        # Analyze each row
        for i, row in enumerate(rows[:26]):  # First 26 rows (A-Z)
            y_avg = sum(c[1] for c in row) / len(row)
            x_first = row[0][0]
            x_last = row[-1][0]

            if len(row) > 1:
                col_gaps = [row[j+1][0] - row[j][0] for j in range(len(row)-1)]
                avg_gap = sum(col_gaps) / len(col_gaps)
            else:
                avg_gap = 0

            letter = chr(ord('A') + i)
            print(f"Row {letter} ({i}): y={y_avg:.0f}, x_first={x_first}, x_last={x_last}, "
                  f"n_cols={len(row)}, avg_gap={avg_gap:.1f}")

        # Calculate precise parameters
        if len(rows) >= 2:
            # Origin from first bubble of first row
            first_row = rows[0]
            first_bubble = first_row[0]
            radius = first_bubble[2]

            origin_x = first_bubble[0] - radius
            origin_y = first_bubble[1] - radius

            # Column gap from first row
            col_gaps = [first_row[j+1][0] - first_row[j][0] for j in range(len(first_row)-1)]
            cols_gap = sum(col_gaps) / len(col_gaps)

            # Row gap from multiple rows
            row_gaps = [rows[i+1][0][1] - rows[i][0][1] for i in range(min(10, len(rows)-1))]
            rows_gap = sum(row_gaps) / len(row_gaps)

            # Bubble size from radius
            bubble_size = int(radius * 2)

            print(f"\n{'='*60}")
            print("CALIBRATED PARAMETERS")
            print(f"{'='*60}")
            print(f"Origin: ({origin_x}, {origin_y})")
            print(f"Column gap: {cols_gap:.1f}")
            print(f"Row gap: {rows_gap:.1f}")
            print(f"Bubble size: ({bubble_size}, {bubble_size})")
            print(f"Number of detected rows: {len(rows)}")
            print(f"Columns in first row: {len(first_row)}")

            # Create visualization with calculated grid
            viz = image.copy()

            # Draw detected circles
            for circle in circle_list:
                cv2.circle(viz, (circle[0], circle[1]), circle[2], (0, 255, 0), 1)

            # Draw calculated grid
            for col in range(25):
                x = int(origin_x + col * cols_gap)
                for row in range(26):
                    y = int(origin_y + row * rows_gap)
                    cv2.rectangle(viz, (x, y), (x + bubble_size, y + bubble_size), (255, 0, 0), 1)

            cv2.imwrite("precise_grid.jpg", viz)
            print(f"\nVisualization saved to: precise_grid.jpg")

            # Also save the name region with circles
            cv2.imwrite("name_region_circles.jpg", name_color)
            print(f"Name region saved to: name_region_circles.jpg")

            return {
                'origin': (origin_x, origin_y),
                'cols_gap': cols_gap,
                'rows_gap': rows_gap,
                'bubble_size': (bubble_size, bubble_size)
            }

    return None


def verify_calibration(image_path: str, origin, cols_gap, rows_gap, bubble_size):
    """Verify calibration by sampling and showing results."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    # Sample first 10 columns
    values = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    for col in range(10):
        x = int(origin[0] + col * cols_gap)

        # Sample all rows
        row_values = []
        for row in range(26):
            y = int(origin[1] + row * rows_gap)
            bw, bh = bubble_size

            if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                bubble = gray[y:y+bh, x:x+bw]
                darkness = np.mean(bubble)
                row_values.append((row, darkness, values[row]))

        if row_values:
            sorted_rows = sorted(row_values, key=lambda x: x[1])
            top3 = sorted_rows[:3]
            print(f"Col {col}: {[(v[2], int(v[1])) for v in top3]}")


def main():
    image_path = "/Volumes/Projects/Projects/omrsheetproj/filledomrsheet/41_629.jpg"

    result = extract_and_analyze_name_region(image_path)

    if result:
        verify_calibration(
            image_path,
            result['origin'],
            result['cols_gap'],
            result['rows_gap'],
            result['bubble_size']
        )


if __name__ == "__main__":
    main()

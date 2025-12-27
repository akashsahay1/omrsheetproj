#!/usr/bin/env python3
"""
Manually find the exact bubble grid coordinates by examining actual filled bubbles.
"""

import cv2
import numpy as np


def find_actual_bubbles(image_path: str, expected_name: str):
    """Find actual filled bubbles by looking for the darkest regions."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print(f"Image: {image_path}")
    print(f"Expected name: {expected_name}")
    print(f"Image shape: {image.shape}")

    # Look at the name grid area more carefully
    # The form appears to have the name bubbles in a specific region

    # Let's scan horizontally to find dark vertical bands (filled bubble columns)
    # Expected first letter is the first character of the name

    first_letter = expected_name[0].upper()
    letter_row = ord(first_letter) - ord('A')  # Row index for the letter
    print(f"First letter '{first_letter}' is at row index {letter_row}")

    # Scan different y positions looking for the characteristic pattern
    # of a bubble grid (alternating dark/light)

    print("\n--- Scanning for bubble grid ---")

    # The name bubble section should be in the upper-left area
    # Let's analyze vertical profiles at different x positions
    for x_test in range(170, 800, 50):
        # Get a vertical slice
        v_slice = gray[680:1400, x_test:x_test+10]
        v_profile = np.mean(v_slice, axis=1)

        # Find local minima (dark regions = potential bubbles)
        minima = []
        for i in range(5, len(v_profile)-5):
            if v_profile[i] < v_profile[i-5] and v_profile[i] < v_profile[i+5]:
                if v_profile[i] < 200:  # Dark enough to be a bubble
                    minima.append((i + 680, v_profile[i]))

        if len(minima) >= 5:
            print(f"\nx={x_test}: Found {len(minima)} potential bubble rows")
            for y, darkness in minima[:10]:
                print(f"  y={y}, darkness={darkness:.1f}")

            # Check spacing between minima
            if len(minima) >= 2:
                gaps = [minima[i+1][0] - minima[i][0] for i in range(min(5, len(minima)-1))]
                print(f"  Row gaps: {gaps}")


def find_bubbles_by_template(image_path: str):
    """Find bubbles using template matching with a circle."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print("\n--- Template matching for bubble outlines ---")

    # Create a template for an empty bubble (light circle with dark outline)
    template_size = 28
    template = np.ones((template_size, template_size), dtype=np.uint8) * 230

    # Draw a circle outline
    cv2.circle(template, (14, 14), 11, 180, 2)

    # Search in the name region
    name_region = gray[680:1400, 150:1100]

    result = cv2.matchTemplate(name_region, template, cv2.TM_CCOEFF_NORMED)

    # Find peaks
    threshold = 0.4
    locations = np.where(result >= threshold)

    if len(locations[0]) > 0:
        points = list(zip(locations[1] + 150, locations[0] + 680))  # x, y in image coords

        # Cluster nearby points
        clusters = []
        for p in points:
            found = False
            for cluster in clusters:
                if abs(p[0] - cluster[0]) < 20 and abs(p[1] - cluster[1]) < 20:
                    found = True
                    break
            if not found:
                clusters.append(p)

        # Sort by y then x
        clusters.sort(key=lambda p: (p[1], p[0]))

        print(f"Found {len(clusters)} potential bubble locations")

        # Group into rows
        rows = []
        current_row = [clusters[0]]
        current_y = clusters[0][1]

        for p in clusters[1:]:
            if abs(p[1] - current_y) < 25:
                current_row.append(p)
            else:
                if len(current_row) >= 5:
                    rows.append(sorted(current_row, key=lambda p: p[0]))
                current_row = [p]
                current_y = p[1]

        if len(current_row) >= 5:
            rows.append(sorted(current_row, key=lambda p: p[0]))

        print(f"Found {len(rows)} rows with 5+ bubbles")

        for i, row in enumerate(rows[:10]):
            print(f"\nRow {i}: y≈{row[0][1]}, {len(row)} bubbles")
            print(f"  First 5 x coords: {[p[0] for p in row[:5]]}")
            if len(row) > 1:
                gaps = [row[j+1][0] - row[j][0] for j in range(min(5, len(row)-1))]
                print(f"  Column gaps: {gaps}")

        return rows


def analyze_filled_bubble_region(image_path: str, expected_name: str):
    """Find where the actual filled bubble should be for the first letter."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    first_letter = expected_name[0].upper()
    row_idx = ord(first_letter) - ord('A')

    print(f"\n--- Looking for filled bubble for '{first_letter}' (row {row_idx}) ---")

    # For each potential first column x position, find the darkest point
    # in each row and see if it matches our expected letter

    best_matches = []

    for col_x in range(180, 300, 5):  # Potential first column positions
        for row_y_base in range(680, 720, 5):  # Potential first row y positions
            for row_gap in range(32, 40, 2):  # Potential row gaps
                # Calculate expected y for the first letter
                expected_y = row_y_base + row_idx * row_gap

                # Sample a region around this point
                if expected_y + 28 < gray.shape[0] and col_x + 28 < gray.shape[1]:
                    region = gray[expected_y:expected_y+28, col_x:col_x+28]
                    darkness = np.mean(region)

                    # Also sample the adjacent rows (should be lighter if this is correct)
                    adjacent_darkness = []
                    for adj_row in [row_idx - 1, row_idx + 1]:
                        adj_y = row_y_base + adj_row * row_gap
                        if 0 <= adj_y < gray.shape[0] - 28:
                            adj_region = gray[adj_y:adj_y+28, col_x:col_x+28]
                            adjacent_darkness.append(np.mean(adj_region))

                    if adjacent_darkness:
                        avg_adjacent = np.mean(adjacent_darkness)
                        contrast = avg_adjacent - darkness

                        if contrast > 10:  # The filled bubble should be darker
                            best_matches.append({
                                'col_x': col_x,
                                'row_y_base': row_y_base,
                                'row_gap': row_gap,
                                'darkness': darkness,
                                'contrast': contrast
                            })

    # Sort by contrast (highest first)
    best_matches.sort(key=lambda m: m['contrast'], reverse=True)

    print("\nTop 10 best matches (highest contrast):")
    for match in best_matches[:10]:
        print(f"  x={match['col_x']}, y_base={match['row_y_base']}, "
              f"gap={match['row_gap']}, dark={match['darkness']:.1f}, contrast={match['contrast']:.1f}")

    if best_matches:
        best = best_matches[0]
        print(f"\nBest match: origin=({best['col_x']}, {best['row_y_base']}), row_gap={best['row_gap']}")

        # Visualize
        viz = image.copy()

        # Draw grid with best parameters
        for col in range(25):
            x = best['col_x'] + col * 36  # Assume col_gap=36
            for row in range(26):
                y = best['row_y_base'] + row * best['row_gap']
                if y + 28 < gray.shape[0] and x + 28 < gray.shape[1]:
                    color = (0, 255, 0) if row == row_idx and col == 0 else (255, 0, 0)
                    cv2.rectangle(viz, (x, y), (x+28, y+28), color, 1)

        cv2.imwrite("best_match_grid.jpg", viz)
        print("Saved: best_match_grid.jpg")

        return best

    return None


def main():
    image_path = "/Volumes/Projects/Projects/omrsheetproj/filledomrsheet/41_629.jpg"
    expected_name = "REYANSH KUMAR"

    find_actual_bubbles(image_path, expected_name)
    rows = find_bubbles_by_template(image_path)
    best = analyze_filled_bubble_region(image_path, expected_name)


if __name__ == "__main__":
    main()

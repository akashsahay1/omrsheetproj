"""
Verify template positions by checking expected filled bubble locations
for the known sample data (DEV MANDAL, Roll: 017, Class: 5, School: 002)
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import cv2
import numpy as np
from src.core.template import TemplateLoader

def check_intensity(gray, x, y, w, h):
    """Check intensity at a position"""
    if x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
        return None, None
    roi = gray[y:y+h, x:x+w]
    mean_val = np.mean(roi)
    dark_ratio = np.sum(roi < 150) / roi.size
    return mean_val, dark_ratio

def main():
    print("=" * 70)
    print("VERIFYING EXPECTED BUBBLE POSITIONS")
    print("Known data: DEV MANDAL, Roll: 017, Class: 5, School: 002")
    print("=" * 70)

    img = cv2.imread(str(project_root / "omrfilled" / "omr-filled.png"))
    if img is None:
        print("Cannot load image")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load template
    loader = TemplateLoader()
    template = loader.load_template("vidya_bharti_2025")

    if template is None:
        print("Cannot load template")
        return

    print(f"\nTemplate: {template.name}")
    print(f"Bubble dimensions: {template.bubble_dimensions}")

    bw, bh = template.bubble_dimensions

    # ========== CHECK NAME GRID ==========
    print("\n" + "=" * 50)
    print("NAME GRID - Expected: D E V   M A N D A L")
    print("=" * 50)

    name_block = template.get_field_block('CandidateName')
    if name_block:
        print(f"Origin: {name_block.origin}")
        print(f"Labels gap: {name_block.labels_gap}")
        print(f"Bubbles gap: {name_block.bubbles_gap}")

        ox, oy = name_block.origin

        # Expected filled positions for "DEV MANDAL"
        # D=3, E=4, V=21, M=12, A=0, N=13, D=3, A=0, L=11
        expected = [
            (1, 'D', 3),   # Column 1, Letter D, row index 3
            (2, 'E', 4),   # Column 2, Letter E, row index 4
            (3, 'V', 21),  # Column 3, Letter V, row index 21
            (5, 'M', 12),  # Column 5, Letter M, row index 12
            (6, 'A', 0),   # Column 6, Letter A, row index 0
            (7, 'N', 13),  # Column 7, Letter N, row index 13
            (8, 'D', 3),   # Column 8, Letter D, row index 3
            (9, 'A', 0),   # Column 9, Letter A, row index 0
            (10, 'L', 11), # Column 10, Letter L, row index 11
        ]

        print("\nExpected filled bubbles:")
        for col, letter, row_idx in expected:
            x = ox + (col - 1) * name_block.labels_gap
            y = oy + row_idx * name_block.bubbles_gap
            mean_val, dark_ratio = check_intensity(gray, x, y, bw, bh)
            is_filled = dark_ratio > 0.25 if dark_ratio else False
            status = "FILLED" if is_filled else "EMPTY"
            mean_str = f"{mean_val:.1f}" if mean_val is not None else "N/A"
            ratio_str = f"{dark_ratio:.2f}" if dark_ratio is not None else "N/A"
            print(f"  Col {col}, {letter} (row {row_idx}): ({x}, {y}) - mean={mean_str}, dark_ratio={ratio_str} {status}")

        # Also check some positions that should be empty
        print("\nExpected EMPTY bubbles (for validation):")
        empty_checks = [
            (1, 'A', 0),   # Column 1, Letter A (should be empty, D is filled)
            (1, 'B', 1),   # Column 1, Letter B (should be empty)
            (4, 'A', 0),   # Column 4, space - should be empty
        ]
        for col, letter, row_idx in empty_checks:
            x = ox + (col - 1) * name_block.labels_gap
            y = oy + row_idx * name_block.bubbles_gap
            mean_val, dark_ratio = check_intensity(gray, x, y, bw, bh)
            is_filled = dark_ratio > 0.25 if dark_ratio else False
            status = "EMPTY" if not is_filled else "FILLED (unexpected!)"
            mean_str = f"{mean_val:.1f}" if mean_val is not None else "N/A"
            ratio_str = f"{dark_ratio:.2f}" if dark_ratio is not None else "N/A"
            print(f"  Col {col}, {letter} (row {row_idx}): ({x}, {y}) - mean={mean_str}, dark_ratio={ratio_str} {status}")

    # ========== CHECK ROLL NUMBER ==========
    print("\n" + "=" * 50)
    print("ROLL NUMBER - Expected: 017")
    print("=" * 50)

    roll_block = template.get_field_block('RollNumber')
    if roll_block:
        print(f"Origin: {roll_block.origin}")
        print(f"Labels gap: {roll_block.labels_gap}")
        print(f"Bubbles gap: {roll_block.bubbles_gap}")

        ox, oy = roll_block.origin

        # Expected: 0, 1, 7
        expected = [
            (1, '0', 0),  # Digit 1 = 0
            (2, '1', 1),  # Digit 2 = 1
            (3, '7', 7),  # Digit 3 = 7
        ]

        print("\nExpected filled bubbles:")
        for col, digit, row_idx in expected:
            x = ox + (col - 1) * roll_block.labels_gap
            y = oy + row_idx * roll_block.bubbles_gap
            mean_val, dark_ratio = check_intensity(gray, x, y, bw, bh)
            is_filled = dark_ratio > 0.25 if dark_ratio else False
            status = "FILLED" if is_filled else "EMPTY"
            mean_str = f"{mean_val:.1f}" if mean_val is not None else "N/A"
            ratio_str = f"{dark_ratio:.2f}" if dark_ratio is not None else "N/A"
            print(f"  Digit {col} = {digit} (row {row_idx}): ({x}, {y}) - mean={mean_str}, dark_ratio={ratio_str} {status}")

    # ========== CHECK CLASS ==========
    print("\n" + "=" * 50)
    print("CLASS - Expected: 5")
    print("=" * 50)

    class_block = template.get_field_block('Class')
    if class_block:
        print(f"Origin: {class_block.origin}")
        ox, oy = class_block.origin

        # Expected: 5
        row_idx = 5
        x, y = ox, oy + row_idx * class_block.bubbles_gap
        mean_val, dark_ratio = check_intensity(gray, x, y, bw, bh)
        is_filled = dark_ratio > 0.25 if dark_ratio else False
        status = "FILLED" if is_filled else "EMPTY"
        mean_str = f"{mean_val:.1f}" if mean_val is not None else "N/A"
        ratio_str = f"{dark_ratio:.2f}" if dark_ratio is not None else "N/A"
        print(f"  Class 5 (row {row_idx}): ({x}, {y}) - mean={mean_str}, dark_ratio={ratio_str} {status}")

    # ========== CHECK SCHOOL CODE ==========
    print("\n" + "=" * 50)
    print("SCHOOL CODE - Expected: 002 (or 0002)")
    print("=" * 50)

    school_block = template.get_field_block('SchoolCode')
    if school_block:
        print(f"Origin: {school_block.origin}")
        ox, oy = school_block.origin

        # Expected: 0, 0, 2 (first 3 digits, 4th might be 0 or empty)
        expected = [
            (1, '0', 0),  # Digit 1 = 0
            (2, '0', 0),  # Digit 2 = 0
            (3, '2', 2),  # Digit 3 = 2
            (4, '0', 0),  # Digit 4 = 0 (or empty)
        ]

        print("\nExpected filled bubbles:")
        for col, digit, row_idx in expected:
            x = ox + (col - 1) * school_block.labels_gap
            y = oy + row_idx * school_block.bubbles_gap
            mean_val, dark_ratio = check_intensity(gray, x, y, bw, bh)
            is_filled = dark_ratio > 0.25 if dark_ratio else False
            status = "FILLED" if is_filled else "EMPTY"
            mean_str = f"{mean_val:.1f}" if mean_val is not None else "N/A"
            ratio_str = f"{dark_ratio:.2f}" if dark_ratio is not None else "N/A"
            print(f"  Digit {col} = {digit} (row {row_idx}): ({x}, {y}) - mean={mean_str}, dark_ratio={ratio_str} {status}")

    # ========== CREATE DEBUG IMAGE ==========
    print("\n" + "=" * 50)
    print("Creating verification debug image...")
    print("=" * 50)

    debug_img = img.copy()

    # Draw expected filled positions in GREEN
    # Draw actual template positions in BLUE

    # Name grid
    if name_block:
        ox, oy = name_block.origin
        expected_name = [(1,3), (2,4), (3,21), (5,12), (6,0), (7,13), (8,3), (9,0), (10,11)]
        for col, row_idx in expected_name:
            x = ox + (col - 1) * name_block.labels_gap
            y = oy + row_idx * name_block.bubbles_gap
            cv2.rectangle(debug_img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)

    cv2.imwrite("verify_positions.png", debug_img)

    scale = 0.35
    small = cv2.resize(debug_img, None, fx=scale, fy=scale)
    cv2.imwrite("verify_positions_small.png", small)

    print("Debug images saved: verify_positions.png, verify_positions_small.png")


if __name__ == "__main__":
    main()

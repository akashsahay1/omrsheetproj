"""
Precisely measure bubble positions by examining the actual filled sheet
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import cv2
import numpy as np

def check_position(img, x, y, w, h, name=""):
    """Check intensity at a position"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Ensure within bounds
    if x < 0 or y < 0 or x + w > gray.shape[1] or y + h > gray.shape[0]:
        return None, 0

    roi = gray[y:y+h, x:x+w]
    mean_val = np.mean(roi)
    dark_ratio = np.sum(roi < 150) / roi.size

    return mean_val, dark_ratio


def find_row_in_column(img, col_x, start_y, end_y, bubble_w, bubble_h, step=26):
    """Scan a column to find filled bubbles"""
    results = []
    y = start_y
    row_idx = 0

    while y < end_y:
        mean_val, dark_ratio = check_position(img, col_x, y, bubble_w, bubble_h)
        if mean_val is not None:
            is_filled = dark_ratio > 0.3 or mean_val < 160
            results.append({
                'row': row_idx,
                'y': y,
                'mean': mean_val,
                'dark_ratio': dark_ratio,
                'filled': is_filled
            })
        y += step
        row_idx += 1

    return results


def analyze_name_grid():
    """Analyze name grid to find exact parameters"""

    print("=" * 70)
    print("NAME GRID ANALYSIS")
    print("=" * 70)

    img = cv2.imread(str(project_root / "omrfilled" / "omr-filled.png"))
    if img is None:
        print("Cannot load image")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try different starting Y positions to find where bubbles are
    print("\nScanning column 1 (x=122) with different start Y and gaps...")

    # Test parameters
    col_x = 122
    bubble_w, bubble_h = 36, 36

    for start_y in [200, 210, 220, 230, 240]:
        for gap in [24, 26, 28, 30]:
            results = find_row_in_column(img, col_x, start_y, 900, bubble_w, bubble_h, gap)
            filled_rows = [r for r in results if r['filled']]

            if filled_rows:
                filled_letters = [chr(65 + r['row']) for r in filled_rows if r['row'] < 26]
                print(f"\nstart_y={start_y}, gap={gap}:")
                print(f"  Filled rows: {[r['row'] for r in filled_rows]}")
                print(f"  Letters: {filled_letters}")

                # If we found 'D', this might be correct
                if 'D' in filled_letters:
                    print(f"  *** Found D! Origin would be ({col_x}, {start_y}), gap={gap}")

    # Also try to find filled bubbles by scanning the whole area
    print("\n\nScanning for ALL dark regions in column 1 area...")
    dark_positions = []

    for y in range(180, 900, 5):
        mean_val, dark_ratio = check_position(img, col_x, y, bubble_w, bubble_h)
        if mean_val is not None and (dark_ratio > 0.35 or mean_val < 150):
            dark_positions.append((y, mean_val, dark_ratio))

    print(f"Found {len(dark_positions)} dark regions at column x={col_x}")
    for y, mean_val, ratio in dark_positions[:10]:
        print(f"  y={y}: mean={mean_val:.1f}, dark_ratio={ratio:.2f}")


def analyze_info_section():
    """Analyze info section to find exact parameters"""

    print("\n" + "=" * 70)
    print("INFO SECTION ANALYSIS (Roll/Class/Section/School)")
    print("=" * 70)

    img = cv2.imread(str(project_root / "omrfilled" / "omr-filled.png"))
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Known: Roll Number is "017"
    # We need to find where digit 0 is at row 0, digit 1 is at row 1, digit 7 is at row 7

    print("\nScanning Roll Number area...")

    # Test different x positions for the 3 roll number columns
    roll_x_positions = [1520, 1530, 1540, 1550, 1560, 1570, 1580, 1590, 1600]

    for col_x in roll_x_positions:
        for start_y in [300, 310, 320, 330, 340]:
            for gap in [24, 26, 28, 30, 32]:
                results = find_row_in_column(img, col_x, start_y, 600, 36, 36, gap)
                filled_rows = [r for r in results if r['filled']]

                if filled_rows:
                    digits = [str(r['row']) for r in filled_rows if r['row'] < 10]
                    if '0' in digits or '1' in digits or '7' in digits:
                        print(f"\nx={col_x}, start_y={start_y}, gap={gap}:")
                        print(f"  Filled digits: {digits}")


def find_actual_bubble_centers():
    """Find actual bubble centers using edge detection"""

    print("\n" + "=" * 70)
    print("FINDING ACTUAL BUBBLE CENTERS")
    print("=" * 70)

    img = cv2.imread(str(project_root / "omrfilled" / "omr-filled.png"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Look for dark filled areas using a stricter threshold
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for bubble-sized shapes
    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 300 < area < 3000:  # Stricter size filter
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            if 0.6 < aspect < 1.6:  # More circular
                cx = x + w // 2
                cy = y + h // 2
                bubbles.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'cx': cx, 'cy': cy, 'area': area
                })

    # Sort by position
    bubbles.sort(key=lambda b: (b['y'], b['x']))

    # Group by region
    name_bubbles = [b for b in bubbles if b['x'] < 1100 and b['y'] < 1100]
    info_bubbles = [b for b in bubbles if b['x'] > 1500 and 250 < b['y'] < 700]

    print(f"\nFOUND {len(name_bubbles)} filled bubbles in NAME region")
    name_bubbles.sort(key=lambda b: (b['x'], b['y']))

    if name_bubbles:
        # Find unique column positions
        x_values = sorted(set(b['cx'] for b in name_bubbles))
        print(f"Unique X centers: {x_values[:15]}...")

        # First column analysis
        first_col_x = x_values[0] if x_values else 0
        first_col = [b for b in name_bubbles if abs(b['cx'] - first_col_x) < 25]
        first_col.sort(key=lambda b: b['y'])

        print(f"\nFirst column (x~{first_col_x}):")
        for i, b in enumerate(first_col):
            print(f"  Bubble {i}: y={b['y']}, cy={b['cy']}, size={b['w']}x{b['h']}")

        if len(first_col) >= 2:
            y_gaps = [first_col[i+1]['cy'] - first_col[i]['cy'] for i in range(len(first_col)-1)]
            print(f"\nY gaps between bubbles: {y_gaps}")
            avg_gap = sum(y_gaps) / len(y_gaps)
            print(f"Average Y gap: {avg_gap:.1f}")

    print(f"\nFOUND {len(info_bubbles)} filled bubbles in INFO region")
    info_bubbles.sort(key=lambda b: b['x'])

    if info_bubbles:
        for i, b in enumerate(info_bubbles):
            print(f"  Bubble {i}: x={b['x']}, y={b['y']}, cx={b['cx']}, cy={b['cy']}")

        # Group by column
        x_values = sorted(set(b['cx'] for b in info_bubbles))
        print(f"\nColumn X centers: {x_values}")


def main():
    analyze_name_grid()
    analyze_info_section()
    find_actual_bubble_centers()


if __name__ == "__main__":
    main()

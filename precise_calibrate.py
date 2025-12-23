"""
Precise calibration tool to find exact bubble coordinates
Analyzes the actual OMR sheet image to find bubble grid positions
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

import cv2
import numpy as np

def find_bubbles_in_region(img, x, y, w, h, min_area=200, max_area=3000):
    """Find all bubble-like circles in a region"""
    roi = img[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Use adaptive threshold for better detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = bw / bh if bh > 0 else 0
            # Bubble-like: roughly circular
            if 0.6 < aspect < 1.6:
                # Convert to absolute coordinates
                abs_x = x + bx + bw // 2  # Center of bubble
                abs_y = y + by + bh // 2
                bubbles.append({
                    'x': x + bx,
                    'y': y + by,
                    'w': bw,
                    'h': bh,
                    'cx': abs_x,
                    'cy': abs_y,
                    'area': area
                })

    return bubbles


def analyze_grid_structure(bubbles, axis='x', tolerance=20):
    """Analyze bubble positions to find grid structure"""
    if not bubbles:
        return []

    # Get positions along the axis
    positions = sorted(set(b['c' + axis] for b in bubbles))

    # Group positions that are close together
    groups = []
    current_group = [positions[0]]

    for pos in positions[1:]:
        if pos - current_group[-1] < tolerance:
            current_group.append(pos)
        else:
            groups.append(int(np.mean(current_group)))
            current_group = [pos]
    groups.append(int(np.mean(current_group)))

    return groups


def find_grid_origin_and_gaps(bubbles, direction='vertical'):
    """Find origin point and gap sizes for a bubble grid"""
    if len(bubbles) < 2:
        return None

    # Get unique x and y positions
    x_positions = analyze_grid_structure(bubbles, 'x', tolerance=15)
    y_positions = analyze_grid_structure(bubbles, 'y', tolerance=15)

    if not x_positions or not y_positions:
        return None

    # Calculate gaps
    x_gaps = [x_positions[i+1] - x_positions[i] for i in range(len(x_positions)-1)] if len(x_positions) > 1 else [0]
    y_gaps = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)] if len(y_positions) > 1 else [0]

    avg_x_gap = int(np.median(x_gaps)) if x_gaps else 0
    avg_y_gap = int(np.median(y_gaps)) if y_gaps else 0

    return {
        'origin': (min(x_positions), min(y_positions)),
        'x_positions': x_positions,
        'y_positions': y_positions,
        'x_gap': avg_x_gap,
        'y_gap': avg_y_gap,
        'num_cols': len(x_positions),
        'num_rows': len(y_positions),
        'avg_bubble_size': int(np.mean([b['w'] for b in bubbles]))
    }


def calibrate_from_image(image_path: str):
    """Calibrate template from actual image"""
    print("=" * 70)
    print("PRECISE TEMPLATE CALIBRATION")
    print("=" * 70)

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Image dimensions: {w} x {h}")

    # ========== CANDIDATE NAME REGION ==========
    print("\n" + "=" * 50)
    print("CANDIDATE NAME GRID (25 columns x 26 rows A-Z)")
    print("=" * 50)

    # The name grid is in the left portion of the sheet, upper area
    # Looking at typical OMR sheets, it's usually in the left 1/3 to 1/2
    name_region = (80, 165, 1050, 900)  # x, y, w, h
    name_bubbles = find_bubbles_in_region(img, *name_region, min_area=150, max_area=1500)
    print(f"Found {len(name_bubbles)} potential bubbles in name region")

    if name_bubbles:
        grid_info = find_grid_origin_and_gaps(name_bubbles)
        if grid_info:
            print(f"  Origin (top-left): {grid_info['origin']}")
            print(f"  Columns (horizontal): {grid_info['num_cols']}")
            print(f"  Rows (vertical): {grid_info['num_rows']}")
            print(f"  X gap (between columns/labels): {grid_info['x_gap']}")
            print(f"  Y gap (between rows/bubbles): {grid_info['y_gap']}")
            print(f"  Bubble size: ~{grid_info['avg_bubble_size']}px")
            print(f"\n  First few X positions: {grid_info['x_positions'][:10]}")
            print(f"  First few Y positions: {grid_info['y_positions'][:10]}")

    # ========== ROLL NUMBER REGION ==========
    print("\n" + "=" * 50)
    print("ROLL NUMBER (3 columns x 10 rows 0-9)")
    print("=" * 50)

    # Roll number is usually in the upper-right area
    roll_region = (1550, 280, 130, 350)
    roll_bubbles = find_bubbles_in_region(img, *roll_region, min_area=150, max_area=1500)
    print(f"Found {len(roll_bubbles)} potential bubbles in roll region")

    if roll_bubbles:
        grid_info = find_grid_origin_and_gaps(roll_bubbles)
        if grid_info:
            print(f"  Origin: {grid_info['origin']}")
            print(f"  Columns: {grid_info['num_cols']}, Rows: {grid_info['num_rows']}")
            print(f"  X gap: {grid_info['x_gap']}, Y gap: {grid_info['y_gap']}")
            print(f"  X positions: {grid_info['x_positions']}")
            print(f"  Y positions: {grid_info['y_positions'][:5]}...")

    # ========== CLASS REGION ==========
    print("\n" + "=" * 50)
    print("CLASS (1 column x 10 rows 0-9)")
    print("=" * 50)

    class_region = (1685, 280, 50, 350)
    class_bubbles = find_bubbles_in_region(img, *class_region, min_area=150, max_area=1500)
    print(f"Found {len(class_bubbles)} potential bubbles in class region")

    if class_bubbles:
        grid_info = find_grid_origin_and_gaps(class_bubbles)
        if grid_info:
            print(f"  Origin: {grid_info['origin']}")
            print(f"  Y positions: {grid_info['y_positions']}")
            print(f"  Y gap: {grid_info['y_gap']}")

    # ========== SECTION REGION ==========
    print("\n" + "=" * 50)
    print("SECTION (1 column x 8 rows A-H)")
    print("=" * 50)

    section_region = (1735, 280, 50, 300)
    section_bubbles = find_bubbles_in_region(img, *section_region, min_area=150, max_area=1500)
    print(f"Found {len(section_bubbles)} potential bubbles in section region")

    if section_bubbles:
        grid_info = find_grid_origin_and_gaps(section_bubbles)
        if grid_info:
            print(f"  Origin: {grid_info['origin']}")
            print(f"  Y positions: {grid_info['y_positions']}")

    # ========== SCHOOL CODE REGION ==========
    print("\n" + "=" * 50)
    print("SCHOOL CODE (4 columns x 10 rows 0-9)")
    print("=" * 50)

    school_region = (1785, 280, 200, 350)
    school_bubbles = find_bubbles_in_region(img, *school_region, min_area=150, max_area=1500)
    print(f"Found {len(school_bubbles)} potential bubbles in school region")

    if school_bubbles:
        grid_info = find_grid_origin_and_gaps(school_bubbles)
        if grid_info:
            print(f"  Origin: {grid_info['origin']}")
            print(f"  X positions: {grid_info['x_positions']}")
            print(f"  Y positions: {grid_info['y_positions'][:5]}...")
            print(f"  X gap: {grid_info['x_gap']}, Y gap: {grid_info['y_gap']}")

    # ========== ANSWERS REGION ==========
    print("\n" + "=" * 50)
    print("ANSWERS (6 columns of 10 questions each, 4 options A-D)")
    print("=" * 50)

    # Questions are in the lower portion
    answers_region = (100, 1440, 2350, 800)
    answer_bubbles = find_bubbles_in_region(img, *answers_region, min_area=200, max_area=2000)
    print(f"Found {len(answer_bubbles)} potential bubbles in answers region")

    if answer_bubbles:
        grid_info = find_grid_origin_and_gaps(answer_bubbles)
        if grid_info:
            print(f"  Origin: {grid_info['origin']}")
            print(f"  First 10 X positions: {grid_info['x_positions'][:10]}")
            print(f"  Y positions: {grid_info['y_positions'][:5]}...")
            print(f"  X gap: {grid_info['x_gap']}, Y gap: {grid_info['y_gap']}")

    # ========== SAVE DEBUG IMAGE ==========
    print("\n" + "=" * 50)
    print("Creating debug visualization...")
    print("=" * 50)

    debug_img = img.copy()

    # Draw detected regions
    regions = [
        (name_region, (0, 255, 0), "NAME"),
        (roll_region, (255, 0, 0), "ROLL"),
        (class_region, (0, 255, 255), "CLASS"),
        (section_region, (255, 0, 255), "SECTION"),
        (school_region, (0, 165, 255), "SCHOOL"),
        (answers_region, (0, 0, 255), "ANSWERS"),
    ]

    for (x, y, w, h), color, label in regions:
        cv2.rectangle(debug_img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(debug_img, label, (x+5, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Draw all detected bubbles
    for bubble in name_bubbles:
        cv2.circle(debug_img, (bubble['cx'], bubble['cy']), 5, (0, 255, 0), -1)

    for bubble in roll_bubbles + class_bubbles + section_bubbles + school_bubbles:
        cv2.circle(debug_img, (bubble['cx'], bubble['cy']), 5, (255, 0, 0), -1)

    for bubble in answer_bubbles:
        cv2.circle(debug_img, (bubble['cx'], bubble['cy']), 5, (0, 0, 255), -1)

    cv2.imwrite("calibration_debug.png", debug_img)

    # Save smaller version
    scale = 0.4
    small = cv2.resize(debug_img, None, fx=scale, fy=scale)
    cv2.imwrite("calibration_debug_small.png", small)

    print(f"Debug images saved: calibration_debug.png, calibration_debug_small.png")


def interactive_point_finder(image_path: str):
    """Open image and let user click to find coordinates"""
    print("\n" + "=" * 50)
    print("MANUAL POINT FINDER")
    print("Click on bubbles to get their coordinates")
    print("Press 'q' to quit")
    print("=" * 50)

    img = cv2.imread(image_path)
    if img is None:
        return

    # Resize for display
    scale = 0.4
    display = cv2.resize(img, None, fx=scale, fy=scale)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert to original coordinates
            orig_x = int(x / scale)
            orig_y = int(y / scale)
            print(f"Click at: ({orig_x}, {orig_y})")

    cv2.namedWindow("OMR Sheet")
    cv2.setMouseCallback("OMR Sheet", mouse_callback)

    while True:
        cv2.imshow("OMR Sheet", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    sample_path = str(project_root / "omrfilled" / "omr-filled.png")

    # Run auto-calibration
    calibrate_from_image(sample_path)

    print("\n" + "=" * 70)
    print("CALIBRATION COMPLETE")
    print("Use the debug images and coordinates above to update template.json")
    print("=" * 70)

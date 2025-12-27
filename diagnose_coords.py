#!/usr/bin/env python3
"""
Diagnostic script to visualize OMR coordinates and debug detection issues.
"""
import cv2
import numpy as np
from pathlib import Path

# Target dimensions
TARGET_WIDTH = 2480
TARGET_HEIGHT = 3507

# Field configurations (from yolo_omr_processor.py)
FIELD_CONFIGS = {
    'name': {
        'origin': (193, 690),
        'bubble_size': (28, 28),
        'rows_gap': 51,
        'cols_gap': 54,
        'rows': 26,
        'cols': 25,
        'values': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
    },
    'roll_number': {
        'origin': (1624, 1278),
        'bubble_size': (28, 28),
        'rows_gap': 54,
        'cols_gap': 56,
        'rows': 10,
        'cols': 3,
        'values': list('0123456789'),
    },
    'class': {
        'origin': (1792, 1278),
        'bubble_size': (28, 28),
        'rows_gap': 54,
        'cols_gap': 56,
        'rows': 8,
        'cols': 1,
        'values': ['3', '4', '5', '6', '7', '8', '9', '10'],
    },
    'section': {
        'origin': (1848, 1278),
        'bubble_size': (28, 28),
        'rows_gap': 54,
        'cols_gap': 56,
        'rows': 8,
        'cols': 1,
        'values': list('ABCDEFGH'),
    },
    'school_code': {
        'origin': (1904, 1278),
        'bubble_size': (28, 28),
        'rows_gap': 54,
        'cols_gap': 56,
        'rows': 10,
        'cols': 3,
        'values': list('0123456789'),
    }
}

ANSWER_BLOCKS = [
    {'origin': (167, 1656), 'questions': range(1, 11)},
    {'origin': (557, 1656), 'questions': range(11, 21)},
    {'origin': (947, 1656), 'questions': range(21, 31)},
    {'origin': (1337, 1656), 'questions': range(31, 41)},
    {'origin': (1727, 1656), 'questions': range(41, 51)},
    {'origin': (2117, 1656), 'questions': range(51, 61)},
]
ANSWER_BUBBLE_SIZE = (28, 28)
ANSWER_ROWS_GAP = 54
ANSWER_COLS_GAP = 70
ANSWER_VALUES = list('ABCD')


def analyze_image(image_path):
    """Analyze a single image and visualize coordinates."""
    print(f"\nAnalyzing: {image_path}")
    print("=" * 60)

    # Load and resize image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image")
        return

    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create visualization copy
    viz = img.copy()

    # Analyze section field specifically
    print("\n--- SECTION FIELD ANALYSIS ---")
    config = FIELD_CONFIGS['section']
    origin = config['origin']
    print(f"Section origin: {origin}")

    for row, val in enumerate(config['values']):
        y = origin[1] + row * config['rows_gap']
        x = origin[0]
        bw, bh = config['bubble_size']

        if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
            bubble = gray[y:y+bh, x:x+bw]
            darkness = np.mean(bubble)
            print(f"  {val}: pos=({x},{y}), darkness={darkness:.1f}")

            # Draw on visualization
            color = (0, 255, 0) if darkness < 200 else (0, 0, 255)
            cv2.rectangle(viz, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(viz, val, (x-20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Analyze first answer block (Q1-Q10)
    print("\n--- ANSWERS BLOCK 1 (Q1-10) ANALYSIS ---")
    block = ANSWER_BLOCKS[0]
    origin = block['origin']
    print(f"Block origin: {origin}")

    for q_idx, q_num in enumerate(list(block['questions'])[:5]):  # First 5 questions
        x = origin[0] + q_idx * ANSWER_COLS_GAP
        print(f"\n  Q{q_num}:")

        for opt_idx, opt in enumerate(ANSWER_VALUES):
            y = origin[1] + opt_idx * ANSWER_ROWS_GAP
            bw, bh = ANSWER_BUBBLE_SIZE

            if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                bubble = gray[y:y+bh, x:x+bw]
                darkness = np.mean(bubble)
                print(f"    {opt}: pos=({x},{y}), darkness={darkness:.1f}")

                # Draw on visualization
                color = (0, 255, 0) if darkness < 200 else (0, 0, 255)
                cv2.rectangle(viz, (x, y), (x+bw, y+bh), color, 1)

    # Look at the actual region around section to see what's there
    print("\n--- SECTION REGION VISUAL CHECK ---")
    section_x = config['origin'][0]
    section_y = config['origin'][1]

    # Check a wider region
    region_x1 = max(0, section_x - 100)
    region_y1 = max(0, section_y - 50)
    region_x2 = min(gray.shape[1], section_x + 200)
    region_y2 = min(gray.shape[0], section_y + 500)

    print(f"Checking region: ({region_x1},{region_y1}) to ({region_x2},{region_y2})")
    region = gray[region_y1:region_y2, region_x1:region_x2]

    # Find dark spots in this region
    dark_mask = region < 150
    dark_coords = np.where(dark_mask)
    if len(dark_coords[0]) > 0:
        print(f"Found {len(dark_coords[0])} dark pixels in section region")
        min_y = np.min(dark_coords[0]) + region_y1
        max_y = np.max(dark_coords[0]) + region_y1
        min_x = np.min(dark_coords[1]) + region_x1
        max_x = np.max(dark_coords[1]) + region_x1
        print(f"Dark region bounds: y=[{min_y},{max_y}], x=[{min_x},{max_x}]")
    else:
        print("No dark pixels found in section region - field may be empty or coords wrong")

    # Save visualization
    output_path = Path(image_path).stem + "_coords_viz.jpg"
    cv2.imwrite(output_path, viz)
    print(f"\nSaved visualization to: {output_path}")

    # Also save a zoomed view of the section/class area
    zoom_x1 = 1600
    zoom_y1 = 1200
    zoom_x2 = 2100
    zoom_y2 = 1900
    zoom = img[zoom_y1:zoom_y2, zoom_x1:zoom_x2].copy()

    # Draw grid on zoom
    for field_name in ['roll_number', 'class', 'section', 'school_code']:
        config = FIELD_CONFIGS[field_name]
        ox, oy = config['origin']
        ox -= zoom_x1
        oy -= zoom_y1

        for col in range(config['cols']):
            for row in range(config['rows']):
                x = ox + col * config['cols_gap']
                y = oy + row * config['rows_gap']
                bw, bh = config['bubble_size']

                if 0 <= x < zoom.shape[1] and 0 <= y < zoom.shape[0]:
                    cv2.rectangle(zoom, (x, y), (x+bw, y+bh), (255, 0, 0), 1)

    zoom_path = Path(image_path).stem + "_zoom_right.jpg"
    cv2.imwrite(zoom_path, zoom)
    print(f"Saved zoomed right side to: {zoom_path}")

    # Save zoomed answers area
    ans_zoom_x1 = 100
    ans_zoom_y1 = 1600
    ans_zoom_x2 = 800
    ans_zoom_y2 = 2000
    ans_zoom = img[ans_zoom_y1:ans_zoom_y2, ans_zoom_x1:ans_zoom_x2].copy()

    # Draw answer grid
    for block in ANSWER_BLOCKS[:2]:  # First 2 blocks
        ox, oy = block['origin']
        ox -= ans_zoom_x1
        oy -= ans_zoom_y1

        for q_idx in range(10):
            for opt_idx in range(4):
                x = ox + q_idx * ANSWER_COLS_GAP
                y = oy + opt_idx * ANSWER_ROWS_GAP
                bw, bh = ANSWER_BUBBLE_SIZE

                if 0 <= x < ans_zoom.shape[1] and 0 <= y < ans_zoom.shape[0]:
                    cv2.rectangle(ans_zoom, (x, y), (x+bw, y+bh), (255, 0, 0), 1)

    ans_zoom_path = Path(image_path).stem + "_zoom_answers.jpg"
    cv2.imwrite(ans_zoom_path, ans_zoom)
    print(f"Saved zoomed answers to: {ans_zoom_path}")


def main():
    # Find a sample image
    images = list(Path("filledomrsheet").glob("*.jpg"))[:1]

    if not images:
        print("No images found in filledomrsheet/")
        return

    for img_path in images:
        analyze_image(img_path)


if __name__ == "__main__":
    main()

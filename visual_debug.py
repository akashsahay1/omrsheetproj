"""
Visual debug tool - draws template coordinates on the image
to show exactly where it's looking for bubbles
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

def draw_template_grid(image_path: str, output_path: str = "visual_debug.png"):
    """Draw template bubble locations on the image"""

    print("=" * 70)
    print("VISUAL DEBUG - Drawing template coordinates on image")
    print("=" * 70)

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot load {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Image size: {w} x {h}")

    # Load template
    loader = TemplateLoader()
    template = loader.load_template("vidya_bharti_2025")

    if template is None:
        print("Error: Cannot load template")
        return

    print(f"Template: {template.name}")
    print(f"Bubble dimensions: {template.bubble_dimensions}")

    # Create a copy for drawing
    debug_img = img.copy()

    # Colors for different field types
    colors = {
        'CandidateName': (0, 255, 0),      # Green
        'RollNumber': (255, 0, 0),          # Blue
        'Class': (0, 255, 255),             # Yellow
        'Section': (255, 0, 255),           # Magenta
        'SchoolCode': (0, 165, 255),        # Orange
        'Answers': (0, 0, 255),             # Red
    }

    # Draw each field block
    for block_name, block in template.field_blocks.items():
        print(f"\n{block_name}:")
        print(f"  Origin: {block.origin}")
        print(f"  Labels gap: {block.labels_gap}")
        print(f"  Bubbles gap: {block.bubbles_gap}")
        print(f"  Direction: {block.direction}")
        print(f"  Labels: {len(block.field_labels)}, Values: {len(block.bubble_values)}")

        # Get color
        if 'Answer' in block_name:
            color = colors['Answers']
        else:
            color = colors.get(block_name, (255, 255, 255))

        # Get all bubble coordinates
        coords = block.get_bubble_coordinates()

        # Draw origin point (big circle)
        ox, oy = block.origin
        cv2.circle(debug_img, (ox, oy), 10, color, -1)
        cv2.putText(debug_img, block_name[:8], (ox, oy - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw all bubble positions
        for i, coord in enumerate(coords):
            x, y = coord['x'], coord['y']
            bw, bh = coord['width'], coord['height']

            # Draw rectangle for bubble area
            cv2.rectangle(debug_img, (x, y), (x + bw, y + bh), color, 1)

            # Draw small center dot
            cv2.circle(debug_img, (x + bw//2, y + bh//2), 2, color, -1)

        # Show first bubble info
        if coords:
            first = coords[0]
            print(f"  First bubble: x={first['x']}, y={first['y']}, "
                  f"label={first['label']}, value={first['value']}")

            # For name grid, show where D, E, V would be
            if block_name == 'CandidateName':
                # D is at index 3 (0-indexed), first column (label_idx=0)
                d_idx = 3  # D
                d_coord = [c for c in coords if c['label'] == 'Name_1' and c['value'] == 'D']
                if d_coord:
                    print(f"  Letter D (col 1) should be at: ({d_coord[0]['x']}, {d_coord[0]['y']})")

    # Save debug image
    cv2.imwrite(output_path, debug_img)
    print(f"\nDebug image saved: {output_path}")

    # Also save scaled version
    scale = 0.35
    small = cv2.resize(debug_img, None, fx=scale, fy=scale)
    small_path = output_path.replace('.png', '_small.png')
    cv2.imwrite(small_path, small)
    print(f"Scaled image saved: {small_path}")


def zoom_region(image_path: str, region_name: str, x: int, y: int, w: int, h: int, output_path: str):
    """Extract and save a zoomed region with template overlay"""

    img = cv2.imread(image_path)
    if img is None:
        return

    # Load template
    loader = TemplateLoader()
    template = loader.load_template("vidya_bharti_2025")

    if template is None:
        return

    # Extract region
    roi = img[y:y+h, x:x+w].copy()

    # Draw bubbles that fall within this region
    for block_name, block in template.field_blocks.items():
        if region_name not in block_name and region_name.lower() not in block_name.lower():
            continue

        coords = block.get_bubble_coordinates()

        for coord in coords:
            bx, by = coord['x'], coord['y']
            bw, bh = coord['width'], coord['height']

            # Check if bubble is in region
            if x <= bx < x + w and y <= by < y + h:
                # Adjust coordinates relative to ROI
                rel_x = bx - x
                rel_y = by - y
                cv2.rectangle(roi, (rel_x, rel_y), (rel_x + bw, rel_y + bh), (0, 0, 255), 2)

    cv2.imwrite(output_path, roi)
    print(f"Zoomed region saved: {output_path}")


def analyze_intensity_at_position(image_path: str, x: int, y: int, w: int, h: int):
    """Check what the intensity is at a specific position"""

    img = cv2.imread(image_path)
    if img is None:
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Extract ROI
    if y + h > gray.shape[0] or x + w > gray.shape[1]:
        print(f"Position ({x}, {y}) with size ({w}, {h}) is out of bounds")
        return

    roi = gray[y:y+h, x:x+w]
    mean_intensity = np.mean(roi)
    min_intensity = np.min(roi)
    max_intensity = np.max(roi)

    # Count dark pixels (filled)
    dark_pixels = np.sum(roi < 150)
    total_pixels = roi.size
    fill_ratio = dark_pixels / total_pixels

    print(f"Position ({x}, {y}) - {w}x{h}:")
    print(f"  Mean intensity: {mean_intensity:.1f}")
    print(f"  Min/Max: {min_intensity}/{max_intensity}")
    print(f"  Fill ratio: {fill_ratio:.2%}")
    print(f"  Likely filled: {mean_intensity < 150 or fill_ratio > 0.3}")


def check_specific_bubbles(image_path: str):
    """Check intensity at specific positions where we know bubbles should be filled"""

    print("\n" + "=" * 70)
    print("CHECKING SPECIFIC BUBBLE POSITIONS")
    print("=" * 70)

    # Load template
    loader = TemplateLoader()
    template = loader.load_template("vidya_bharti_2025")

    if template is None:
        print("Error: Cannot load template")
        return

    # For the name "DEV MANDAL":
    # Column 1 should have D filled (index 3)
    # Column 2 should have E filled (index 4)
    # Column 3 should have V filled (index 21)

    print("\nExpected filled positions for name 'DEV':")

    name_block = template.get_field_block('CandidateName')
    if name_block:
        coords = name_block.get_bubble_coordinates()

        # Find where D should be (Name_1, value D)
        for coord in coords:
            if coord['label'] == 'Name_1' and coord['value'] == 'D':
                print(f"  Column 1, Letter D: ({coord['x']}, {coord['y']})")
                analyze_intensity_at_position(image_path, coord['x'], coord['y'],
                                             coord['width'], coord['height'])

            if coord['label'] == 'Name_2' and coord['value'] == 'E':
                print(f"  Column 2, Letter E: ({coord['x']}, {coord['y']})")
                analyze_intensity_at_position(image_path, coord['x'], coord['y'],
                                             coord['width'], coord['height'])

            if coord['label'] == 'Name_3' and coord['value'] == 'V':
                print(f"  Column 3, Letter V: ({coord['x']}, {coord['y']})")
                analyze_intensity_at_position(image_path, coord['x'], coord['y'],
                                             coord['width'], coord['height'])

    # For roll number "017":
    # Digit 1: 0, Digit 2: 1, Digit 3: 7
    print("\nExpected filled positions for roll '017':")

    roll_block = template.get_field_block('RollNumber')
    if roll_block:
        coords = roll_block.get_bubble_coordinates()

        for coord in coords:
            if coord['label'] == 'Roll_D1' and coord['value'] == '0':
                print(f"  Digit 1, value 0: ({coord['x']}, {coord['y']})")
                analyze_intensity_at_position(image_path, coord['x'], coord['y'],
                                             coord['width'], coord['height'])

            if coord['label'] == 'Roll_D2' and coord['value'] == '1':
                print(f"  Digit 2, value 1: ({coord['x']}, {coord['y']})")
                analyze_intensity_at_position(image_path, coord['x'], coord['y'],
                                             coord['width'], coord['height'])

            if coord['label'] == 'Roll_D3' and coord['value'] == '7':
                print(f"  Digit 3, value 7: ({coord['x']}, {coord['y']})")
                analyze_intensity_at_position(image_path, coord['x'], coord['y'],
                                             coord['width'], coord['height'])

    # For class "5":
    print("\nExpected filled position for class '5':")

    class_block = template.get_field_block('Class')
    if class_block:
        coords = class_block.get_bubble_coordinates()

        for coord in coords:
            if coord['value'] == '5':
                print(f"  Class value 5: ({coord['x']}, {coord['y']})")
                analyze_intensity_at_position(image_path, coord['x'], coord['y'],
                                             coord['width'], coord['height'])


def main():
    sample_path = str(project_root / "omrfilled" / "omr-filled.png")

    # Draw full template overlay
    draw_template_grid(sample_path, "visual_debug.png")

    # Check specific bubble positions
    check_specific_bubbles(sample_path)

    # Create zoomed views of specific regions
    print("\n" + "=" * 70)
    print("Creating zoomed region images...")
    print("=" * 70)

    zoom_region(sample_path, "Name", 80, 180, 400, 300, "zoom_name.png")
    zoom_region(sample_path, "Roll", 1550, 300, 250, 350, "zoom_roll.png")


if __name__ == "__main__":
    main()

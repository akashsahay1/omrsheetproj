"""Test name detection on multiple sample images."""
import cv2
import numpy as np
import re
import os

# Template parameters (corrected)
template_width, template_height = 2480, 3507
origin_x, origin_y = 193, 690
bubbles_gap = 51
labels_gap = 54
bubble_w, bubble_h = 28, 28

def detect_name(image_path):
    """Detect name from an OMR image."""
    img = cv2.imread(image_path)
    if img is None:
        return None, "Failed to load image"

    img = cv2.resize(img, (template_width, template_height))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detected_chars = []

    for col in range(25):
        x = origin_x + col * labels_gap

        bubble_values = []
        for row in range(26):
            y = origin_y + row * bubbles_gap
            letter = chr(65 + row)

            if 0 < y < template_height - bubble_h and 0 < x < template_width - bubble_w:
                bubble = gray[y:y+bubble_h, x:x+bubble_w]
                darkness = np.mean(bubble)
                bubble_values.append((letter, darkness))

        if bubble_values:
            darkest = min(bubble_values, key=lambda x: x[1])
            vals = [v[1] for v in bubble_values]
            median_val = np.median(vals)

            # Detection threshold: 75% of median
            if darkest[1] < median_val * 0.75:
                detected_chars.append(darkest[0])
            else:
                detected_chars.append(' ')

    raw_name = ''.join(detected_chars)
    name = raw_name.rstrip()
    name = re.sub(r'\s+', ' ', name)

    return name, None

# Get sample images
image_dir = "filledomrsheet"
images = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])[:15]

print("Testing name detection on 15 sample images...")
print("="*70)
print(f"{'Filename':<20} {'Detected Name':<40}")
print("-"*70)

for img_file in images:
    image_path = os.path.join(image_dir, img_file)
    name, error = detect_name(image_path)

    if error:
        print(f"{img_file:<20} ERROR: {error}")
    else:
        print(f"{img_file:<20} {name:<40}")

print("\n" + "="*70)
print("Please verify these detected names against the actual OMR sheets.")

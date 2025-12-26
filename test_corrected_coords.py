"""Test with corrected coordinates based on bubble scanning."""
import cv2
import numpy as np
import re

# Load and resize to template dimensions
img = cv2.imread("filledomrsheet/41_624.jpg")
template_width, template_height = 2480, 3507
img = cv2.resize(img, (template_width, template_height))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Corrected coordinates based on scanning
origin_x = 192  # First column x
origin_y = 690  # First row (A) y
bubbles_gap = 51  # Vertical gap between rows
labels_gap = 54   # Horizontal gap between columns
bubble_w, bubble_h = 28, 28

print("Testing with CORRECTED coordinates:")
print(f"  Origin: ({origin_x}, {origin_y})")
print(f"  bubblesGap: {bubbles_gap}")
print(f"  labelsGap: {labels_gap}")
print("="*60)

# Extract name
detected_chars = []
output = img.copy()

for col in range(25):
    x = origin_x + col * labels_gap

    # Find darkest row in this column
    bubble_values = []
    for row in range(26):
        y = origin_y + row * bubbles_gap
        letter = chr(65 + row)

        if 0 < y < template_height - bubble_h and 0 < x < template_width - bubble_w:
            bubble = gray[y:y+bubble_h, x:x+bubble_w]
            darkness = np.mean(bubble)
            bubble_values.append((letter, darkness, y))

            # Draw rectangle
            color = (0, 255, 0)  # Green default
            if darkness < 100:
                color = (0, 0, 255)  # Red for filled
            cv2.rectangle(output, (x, y), (x+bubble_w, y+bubble_h), color, 1)

    # Find darkest bubble
    if bubble_values:
        darkest = min(bubble_values, key=lambda x: x[1])
        vals = [v[1] for v in bubble_values]
        median_val = np.median(vals)

        # Detection threshold: 75% of median
        if darkest[1] < median_val * 0.75:
            detected_chars.append(darkest[0])
            # Highlight detected bubble
            y = darkest[2]
            cv2.rectangle(output, (x, y), (x+bubble_w, y+bubble_h), (255, 0, 0), 3)
        else:
            detected_chars.append(' ')

    if col < 12:
        print(f"Col {col+1:2d}: detected='{detected_chars[-1]}' "
              f"(darkest={darkest[0]}:{darkest[1]:.0f}, median={median_val:.0f})")

# Assemble name
raw_name = ''.join(detected_chars)
name = raw_name.rstrip()
name = re.sub(r'\s+', ' ', name)

print("\n" + "="*60)
print(f"Detected name: '{name}'")
print(f"Expected name: 'PARI KUMARI'")
print(f"Match: {'YES!' if name == 'PARI KUMARI' else 'NO'}")

# Save visualization
name_area = output[600:2100, 100:1600]
cv2.imwrite("test_corrected_result.jpg", name_area)
print("\nSaved test_corrected_result.jpg")

# If detection worked, print the template values to use
if name.strip():
    print("\n" + "="*60)
    print("TEMPLATE VALUES TO USE:")
    print(f'  "origin": [{origin_x}, {origin_y}],')
    print(f'  "bubblesGap": {bubbles_gap},')
    print(f'  "labelsGap": {labels_gap}')

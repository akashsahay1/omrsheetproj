"""Test relative threshold detection for name bubbles."""
import cv2
import numpy as np
import re

img = cv2.imread('filledomrsheet/41_624.jpg')
img = cv2.resize(img, (1360, 1928))

name_x_origin = 105
name_y_origin = 387
bubble_gap_h = 28
bubble_gap_v = 31
bubble_size = 14
num_columns = 25

print('Detecting name using DARKEST bubble per column:')
print('=' * 60)

detected_name = []
for col in range(num_columns):
    x = name_x_origin + col * bubble_gap_v

    # Get all bubble values for this column
    bubble_values = []
    for row in range(26):
        y = name_y_origin + row * bubble_gap_h
        bubble = img[y:y+bubble_size, x:x+bubble_size]
        gray = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        bubble_values.append((chr(65+row), mean_val))

    # Find darkest
    darkest = min(bubble_values, key=lambda x: x[1])

    # Check if it's significantly darker than median (indicates filled)
    vals = [v for l,v in bubble_values]
    median_val = np.median(vals)

    if darkest[1] < median_val * 0.5:  # At least 50% darker than median
        detected_name.append(darkest[0])
        print(f'Col {col+1:2d}: {darkest[0]} (darkness={darkest[1]:.1f}, median={median_val:.1f})')
    else:
        detected_name.append(' ')  # No clear filled bubble - space
        print(f'Col {col+1:2d}: [EMPTY] (darkest={darkest[0]}:{darkest[1]:.1f}, median={median_val:.1f})')

name = ''.join(detected_name).strip()
# Clean multiple spaces
name = re.sub(r'\s+', ' ', name)
print(f'\nDetected name: "{name}"')
print(f'Expected name: "PARI KUMARI"')

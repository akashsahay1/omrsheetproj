"""Final test with correct coordinates and threshold."""
import cv2
import numpy as np
import re

img = cv2.imread("filledomrsheet/41_624.jpg")
img = cv2.resize(img, (1360, 1928))

# CORRECTED coordinates
name_x_origin = 113
name_y_origin = 386
bubble_gap_h = 28
bubble_gap_v = 30
bubble_size = 18

print("Final detection test:")
print("=" * 60)

detected_name = []

for col in range(25):
    x = name_x_origin + col * bubble_gap_v

    bubble_values = []
    for row in range(26):
        y = name_y_origin + row * bubble_gap_h
        bubble = img[y:y+bubble_size, x:x+bubble_size]
        gray = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        bubble_values.append((chr(65+row), mean_val))

    darkest = min(bubble_values, key=lambda x: x[1])
    vals = [v for l,v in bubble_values]
    median_val = np.median(vals)
    second_darkest = sorted(bubble_values, key=lambda x: x[1])[1]

    # Use 75% threshold AND check gap to second darkest
    gap_ratio = second_darkest[1] / darkest[1] if darkest[1] > 0 else 1

    if darkest[1] < median_val * 0.75 and gap_ratio > 1.1:  # At least 10% darker than 2nd
        detected_name.append(darkest[0])
        print(f"Col {col+1:2d}: {darkest[0]} (val={darkest[1]:.0f}, 2nd={second_darkest[0]}:{second_darkest[1]:.0f}, gap={gap_ratio:.2f})")
    else:
        detected_name.append(' ')
        if col < 15:  # Only print first 15 for brevity
            print(f"Col {col+1:2d}: [SPACE] (darkest={darkest[0]}:{darkest[1]:.0f}, gap={gap_ratio:.2f})")

name = ''.join(detected_name).strip()
name = re.sub(r'\s+', ' ', name)
print(f"\nDetected: '{name}'")
print(f"Expected: 'PARI KUMARI'")

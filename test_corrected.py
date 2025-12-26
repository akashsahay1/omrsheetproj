"""Test with corrected coordinates."""
import cv2
import numpy as np
import re

img = cv2.imread("filledomrsheet/41_624.jpg")
img = cv2.resize(img, (1360, 1928))

# CORRECTED coordinates
name_x_origin = 113  # Was 105, corrected to 113
name_y_origin = 386  # Was 387, corrected to 386
bubble_gap_h = 28    # Vertical gap (rows)
bubble_gap_v = 30    # Horizontal gap (columns) - might also need adjustment
bubble_size = 18     # Slightly larger to capture full bubble

print("Testing with CORRECTED coordinates:")
print(f"Origin: ({name_x_origin}, {name_y_origin})")
print(f"Gaps: h={bubble_gap_h}, v={bubble_gap_v}")
print("=" * 60)

# Visualize first 5 columns
output = img.copy()
detected_name = []

for col in range(10):  # Check first 10 columns
    x = name_x_origin + col * bubble_gap_v

    # Get all bubble values for this column
    bubble_values = []
    for row in range(26):
        y = name_y_origin + row * bubble_gap_h
        bubble = img[y:y+bubble_size, x:x+bubble_size]
        gray = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        bubble_values.append((chr(65+row), mean_val))

        # Draw rectangle on output
        if col < 5:
            color = (0, 255, 0)
            cv2.rectangle(output, (x, y), (x+bubble_size, y+bubble_size), color, 1)

    # Find darkest
    darkest = min(bubble_values, key=lambda x: x[1])
    vals = [v for l,v in bubble_values]
    median_val = np.median(vals)

    if darkest[1] < median_val * 0.5:
        detected_name.append(darkest[0])
        status = f"{darkest[0]} ({darkest[1]:.0f})"
    else:
        detected_name.append(' ')
        status = f"[space] (darkest:{darkest[0]}={darkest[1]:.0f})"

    print(f"Col {col+1:2d}: {status}")

# Save visualization
name_area = output[350:800, 80:350]
cv2.imwrite("test_corrected_alignment.jpg", name_area)

name = ''.join(detected_name).strip()
name = re.sub(r'\s+', ' ', name)
print(f"\nDetected: '{name}'")
print(f"Expected: 'PARI KUMARI'")

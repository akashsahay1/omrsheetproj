"""Test detection on aligned image with calibrated coordinates."""
import cv2
import numpy as np

# Load aligned image
aligned = cv2.imread("aligned_result.jpg")
if aligned is None:
    print("No aligned image found. Run test_alignment.py first.")
    exit(1)

gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
print(f"Aligned image shape: {aligned.shape}")

# Calibrated coordinates for aligned image
# Based on bubble scanning results
origin_x, origin_y = 230, 690  # Adjusted from calibration
rows_gap = 51
cols_gap = 54  # Approximate based on column analysis
bubble_w, bubble_h = 28, 28

print(f"\nUsing coordinates:")
print(f"  Origin: ({origin_x}, {origin_y})")
print(f"  Gaps: rows={rows_gap}, cols={cols_gap}")
print("="*60)

# Detect name
detected_name = []
for col in range(12):  # First 12 columns for "PARI KUMARI "
    x = origin_x + col * cols_gap

    bubble_values = []
    for row in range(26):
        y = origin_y + row * rows_gap
        letter = chr(65 + row)

        if 0 <= y < gray.shape[0] - bubble_h and 0 <= x < gray.shape[1] - bubble_w:
            bubble = gray[y:y+bubble_h, x:x+bubble_w]
            darkness = np.mean(bubble)
            bubble_values.append((letter, darkness))

    if bubble_values:
        darkest = min(bubble_values, key=lambda x: x[1])
        vals = [v[1] for v in bubble_values]
        median_val = np.median(vals)

        if darkest[1] < median_val * 0.75:
            detected_name.append(darkest[0])
            print(f"Col {col+1:2d}: {darkest[0]} (darkness={darkest[1]:.0f}, median={median_val:.0f})")
        else:
            detected_name.append(' ')
            print(f"Col {col+1:2d}: [SPACE] (darkest={darkest[0]}:{darkest[1]:.0f}, median={median_val:.0f})")

name = ''.join(detected_name).strip()
name = ' '.join(name.split())

print("\n" + "="*60)
print(f"Detected name: '{name}'")
print(f"Expected:      'PARI KUMARI'")
print(f"Match: {'YES!' if name == 'PARI KUMARI' else 'NO'}")

# Visualize the grid
output = aligned.copy()
for col in range(12):
    x = origin_x + col * cols_gap
    for row in range(26):
        y = origin_y + row * rows_gap
        if 0 <= y < gray.shape[0] - bubble_h and 0 <= x < gray.shape[1] - bubble_w:
            bubble = gray[y:y+bubble_h, x:x+bubble_w]
            darkness = np.mean(bubble)

            color = (0, 255, 0) if darkness > 150 else (0, 0, 255)
            cv2.rectangle(output, (x, y), (x+bubble_w, y+bubble_h), color, 1)

# Save visualization
cv2.imwrite("detection_grid.jpg", output[600:1800, 150:900])
print("\nSaved detection_grid.jpg")

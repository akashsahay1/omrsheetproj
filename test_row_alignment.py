"""Check row alignment by finding the filled bubble in column 1."""
import cv2
import numpy as np

img = cv2.imread("filledomrsheet/41_624.jpg")
img = cv2.resize(img, (1360, 1928))

# Current settings
name_x_origin = 113
name_y_origin = 386
bubble_gap_h = 28
bubble_size = 18

print("Column 1 analysis - finding the filled bubble:")
print("=" * 60)

# Check each row in column 1
for row in range(26):
    y = name_y_origin + row * bubble_gap_h
    bubble = img[y:y+bubble_size, name_x_origin:name_x_origin+bubble_size]
    gray = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    letter = chr(65 + row)

    # Check if this looks filled (significantly darker)
    marker = ""
    if mean_val < 130:
        marker = " <-- DARK (filled?)"
    elif mean_val < 160:
        marker = " <-- medium"

    print(f"Row {letter} (y={y}): {mean_val:.1f}{marker}")

# Now let's try different y-origins to find better alignment
print("\n\nTrying different y-origins for row P (should be filled in col 1):")
print("=" * 60)
for test_y in range(380, 400, 2):
    # Row P is 15th row (0-indexed), so y = test_y + 15*28
    p_row_y = test_y + 15 * 28
    bubble = img[p_row_y:p_row_y+bubble_size, name_x_origin:name_x_origin+bubble_size]
    gray = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    print(f"y_origin={test_y}, P at y={p_row_y}: darkness={mean_val:.1f}")

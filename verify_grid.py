"""Verify grid parameters by looking at known positions."""
import cv2
import numpy as np

# Load and resize to template dimensions
img = cv2.imread("filledomrsheet/41_624.jpg")
template_width, template_height = 2480, 3507
img = cv2.resize(img, (template_width, template_height))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Based on analysis:
# - Filled bubbles for "PARI KUMARI" are at specific (x,y) positions
# - Need to find exact column 1 x position and column gap

# Known filled positions from analysis:
# P at (195, 1460), A at (245, 690), R at (245, 1565)?, I at (245, 1100)?
# etc.

# Let's trace column by column
print("Tracing actual bubble columns...")
print("="*60)

# For column 1 (P), we found filled at x~195, y~1460
# Let's find the exact center of this bubble
def find_bubble_center(gray, start_x, start_y, search_range=30):
    """Find the darkest spot near the given position."""
    best_x, best_y = start_x, start_y
    best_darkness = 255

    for dx in range(-search_range, search_range+1, 2):
        for dy in range(-search_range, search_range+1, 2):
            x, y = start_x + dx, start_y + dy
            if 0 < y < gray.shape[0]-28 and 0 < x < gray.shape[1]-28:
                bubble = gray[y:y+28, x:x+28]
                darkness = np.mean(bubble)
                if darkness < best_darkness:
                    best_darkness = darkness
                    best_x, best_y = x, y

    return best_x, best_y, best_darkness

# Find centers for each expected letter in "PARI KUMARI"
expected_letters = [
    ('P', 15, 195, 1460),  # Col 1: P at row 15
    ('A', 0, 245, 690),    # Col 2: A at row 0
    ('R', 17, None, None), # Col 3: R at row 17
    ('I', 8, None, None),  # Col 4: I at row 8
]

# Use found P and A to calculate grid parameters
p_x, p_y, p_d = find_bubble_center(gray, 195, 1460)
a_x, a_y, a_d = find_bubble_center(gray, 245, 690)

print(f"P bubble (Col 1, Row 15): x={p_x}, y={p_y}, darkness={p_d:.1f}")
print(f"A bubble (Col 2, Row 0):  x={a_x}, y={a_y}, darkness={a_d:.1f}")

# Calculate parameters
labels_gap = a_x - p_x
print(f"\nCalculated labels_gap (column gap): {labels_gap}")

# Calculate row gap from P (row 15) to A (row 0)
# A should be 15 rows above P
bubbles_gap_from_pa = (p_y - a_y) / 15
print(f"Calculated bubblesGap from P-A (15 rows): {bubbles_gap_from_pa:.1f}")

# Find more letters to verify
print("\n" + "="*60)
print("Verifying with additional letters...")

# Expected positions based on calculated gaps
origin_x = p_x  # Column 1 at P's x
origin_y = a_y  # Row A at A's y

# Verify each letter of "PARI KUMARI"
name = "PARI KUMARI"
print(f"\nExpected: '{name}'")
print("Verifying each column:")

for col, letter in enumerate(name):
    if letter == ' ':
        print(f"  Col {col+1}: [SPACE]")
        continue

    expected_x = origin_x + col * labels_gap
    row_idx = ord(letter) - ord('A')
    expected_y = origin_y + row_idx * bubbles_gap_from_pa

    # Find actual bubble near expected position
    actual_x, actual_y, darkness = find_bubble_center(gray, int(expected_x), int(expected_y))

    # Check if dark enough to be filled
    is_filled = darkness < 80

    print(f"  Col {col+1}: Expected {letter} at ({expected_x:.0f},{expected_y:.0f}) -> "
          f"Found at ({actual_x},{actual_y}), darkness={darkness:.0f}, filled={is_filled}")

# Create visualization
output = img.copy()

# Draw grid based on calculated parameters
for col in range(11):
    x = int(origin_x + col * labels_gap)
    for row in range(26):
        y = int(origin_y + row * bubbles_gap_from_pa)
        if 0 < y < template_height - 28 and 0 < x < template_width - 28:
            bubble = gray[y:y+28, x:x+28]
            darkness = np.mean(bubble)

            color = (0, 255, 0) if darkness >= 80 else (0, 0, 255)
            cv2.rectangle(output, (x, y), (x+28, y+28), color, 1)

# Save
name_area = output[600:2000, 150:900]
cv2.imwrite("verify_grid.jpg", name_area)
print(f"\nSaved verify_grid.jpg")

# Output final template values
print("\n" + "="*60)
print("TEMPLATE VALUES:")
print(f'  "origin": [{origin_x}, {int(origin_y)}],')
print(f'  "bubblesGap": {int(round(bubbles_gap_from_pa))},')
print(f'  "labelsGap": {labels_gap}')

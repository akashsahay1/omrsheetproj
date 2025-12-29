"""Test different grid origin positions to find better alignment."""
import cv2
import numpy as np

image_path = r'F:\Projects\omrsheetprojm\traindata\Khoj\10151A.jpg'
image = cv2.imread(image_path)
aligned = cv2.resize(image, (2480, 3507))
gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

bubble_size = (33, 33)
rows_gap = 52
cols_gap = 54
values = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Current origin
current_origin = (189, 671)

# Test different y-offsets for column 6
col = 6
print("Testing different Y origins for Column 6 (expecting A to be darkest):")
print("=" * 70)
print(f"{'Y Offset':<12} {'Row A dark':<12} {'Row B dark':<12} {'A darker?':<12}")
print("-" * 70)

for y_offset in range(-20, 21, 5):  # Test -20 to +20 pixels
    test_origin_y = current_origin[1] + y_offset
    x = current_origin[0] + col * cols_gap

    # Get darkness for rows A and B
    y_a = test_origin_y + 0 * rows_gap
    y_b = test_origin_y + 1 * rows_gap

    bubble_a = gray[y_a:y_a+bubble_size[1], x:x+bubble_size[0]]
    bubble_b = gray[y_b:y_b+bubble_size[1], x:x+bubble_size[0]]

    dark_a = np.mean(bubble_a)
    dark_b = np.mean(bubble_b)

    a_darker = "YES" if dark_a < dark_b else "NO"
    marker = " <-- BEST" if dark_a < dark_b and dark_a < 200 else ""

    print(f"{y_offset:+4}         {dark_a:<12.1f} {dark_b:<12.1f} {a_darker:<12}{marker}")

# Now test with a much larger range
print("\n" + "=" * 70)
print("Wider search range:")
print("=" * 70)

best_offset = 0
best_gap = 0
for y_offset in range(-50, 51, 2):
    test_origin_y = current_origin[1] + y_offset
    x = current_origin[0] + col * cols_gap

    y_a = test_origin_y + 0 * rows_gap
    y_b = test_origin_y + 1 * rows_gap

    if y_a < 0 or y_b + bubble_size[1] >= gray.shape[0]:
        continue

    bubble_a = gray[y_a:y_a+bubble_size[1], x:x+bubble_size[0]]
    bubble_b = gray[y_b:y_b+bubble_size[1], x:x+bubble_size[0]]

    dark_a = np.mean(bubble_a)
    dark_b = np.mean(bubble_b)

    # We want A to be darker than B by maximum gap
    gap = dark_b - dark_a
    if gap > best_gap:
        best_gap = gap
        best_offset = y_offset

print(f"Best Y offset: {best_offset:+d} (A-B gap: {best_gap:.1f})")

# Apply best offset and show results
test_origin_y = current_origin[1] + best_offset
x = current_origin[0] + col * cols_gap
y_a = test_origin_y + 0 * rows_gap
y_b = test_origin_y + 1 * rows_gap

bubble_a = gray[y_a:y_a+bubble_size[1], x:x+bubble_size[0]]
bubble_b = gray[y_b:y_b+bubble_size[1], x:x+bubble_size[0]]

print(f"\nWith offset {best_offset:+d}:")
print(f"  Row A darkness: {np.mean(bubble_a):.1f}")
print(f"  Row B darkness: {np.mean(bubble_b):.1f}")

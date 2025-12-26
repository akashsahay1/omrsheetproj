"""Fine-tune origin coordinates by testing multiple offsets."""
import cv2
import numpy as np
import re

# Load and resize to template dimensions
img = cv2.imread("filledomrsheet/41_624.jpg")
template_width, template_height = 2480, 3507
img = cv2.resize(img, (template_width, template_height))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

bubbles_gap = 51  # Vertical gap between rows
labels_gap = 54   # Horizontal gap between columns
bubble_w, bubble_h = 28, 28

print("Fine-tuning origin coordinates...")
print("Expected name: 'PARI KUMARI'")
print("="*60)

# Try different x_origin and y_origin values
results = []

for x_offset in range(-30, 31, 5):
    for y_offset in range(-30, 31, 5):
        origin_x = 192 + x_offset
        origin_y = 690 + y_offset

        # Extract first 11 characters
        detected_chars = []
        for col in range(11):
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

                if darkest[1] < median_val * 0.75:
                    detected_chars.append(darkest[0])
                else:
                    detected_chars.append(' ')

        detected = ''.join(detected_chars).strip()
        detected = re.sub(r'\s+', ' ', detected)

        if detected == "PARI KUMARI":
            results.append((x_offset, y_offset, detected, 100))
            print(f"PERFECT MATCH at offset ({x_offset}, {y_offset})")
        elif "PARI" in detected and "KUMARI" in detected:
            score = 90
            results.append((x_offset, y_offset, detected, score))
        elif detected.startswith("PARI"):
            score = 50
            results.append((x_offset, y_offset, detected, score))

# Show best results
print("\n" + "="*60)
print("Best results:")
results.sort(key=lambda x: -x[3])
for x_off, y_off, detected, score in results[:10]:
    print(f"  Offset ({x_off:+3d}, {y_off:+3d}): '{detected}' (score: {score})")

# If no good results, show what current offset gives at different positions
if not results or results[0][3] < 90:
    print("\n" + "="*60)
    print("Detailed analysis of current settings...")

    origin_x, origin_y = 192, 690

    # Check column 1 darkness at different rows
    print("\nColumn 1 (x=192) bubble darkness by row:")
    for row in range(26):
        y = origin_y + row * bubbles_gap
        letter = chr(65 + row)
        x = origin_x

        if 0 < y < template_height - bubble_h:
            bubble = gray[y:y+bubble_h, x:x+bubble_w]
            darkness = np.mean(bubble)
            if darkness < 100:
                print(f"  Row {letter} (y={y}): {darkness:.1f} <-- FILLED")
            elif darkness < 150:
                print(f"  Row {letter} (y={y}): {darkness:.1f} <-- partial")

    # Check first 3 columns to see what's actually being detected
    print("\nFirst 3 columns darkest values:")
    for col in range(3):
        x = origin_x + col * labels_gap
        print(f"\n  Column {col+1} (x={x}):")

        filled_rows = []
        for row in range(26):
            y = origin_y + row * bubbles_gap
            letter = chr(65 + row)

            if 0 < y < template_height - bubble_h:
                bubble = gray[y:y+bubble_h, x:x+bubble_w]
                darkness = np.mean(bubble)
                if darkness < 100:
                    filled_rows.append((letter, darkness, y))

        for letter, darkness, y in sorted(filled_rows, key=lambda x: x[1])[:3]:
            print(f"    {letter} at y={y}: darkness={darkness:.1f}")

# Check if we need to shift y_origin based on where P actually is
print("\n" + "="*60)
print("Searching for P in column 1 area...")

# Scan the area where P should be (if detected as A at y=690, P might be at different offset)
for test_y in range(660, 720, 2):
    for test_x in range(180, 210, 2):
        bubble = gray[test_y:test_y+bubble_h, test_x:test_x+bubble_w]
        darkness = np.mean(bubble)
        if darkness < 50:
            print(f"  Very dark bubble at x={test_x}, y={test_y}: darkness={darkness:.1f}")

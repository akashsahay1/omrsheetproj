"""Verify bubble coordinates by visualizing the grid overlay."""
import cv2
import numpy as np

# Load and resize to template dimensions
img = cv2.imread("filledomrsheet/41_624.jpg")
template_width, template_height = 2480, 3507
img = cv2.resize(img, (template_width, template_height))

# Test multiple origin configurations
configs = [
    # (name, origin_x, origin_y, bubblesGap, labelsGap)
    ("Current template", 186, 477, 28, 21),
    ("HoughCircles suggestion", 192, 742, 51, 54),
    ("Adjusted based on P detection", 192, 700, 51, 54),
]

bubble_w, bubble_h = 28, 28

for config_name, origin_x, origin_y, bubbles_gap, labels_gap in configs:
    output = img.copy()

    # Draw grid for first 5 columns and 26 rows
    for col in range(5):
        x = origin_x + col * labels_gap
        for row in range(26):
            y = origin_y + row * bubbles_gap
            letter = chr(65 + row)

            # Extract bubble region and calculate darkness
            if 0 < y < template_height - bubble_h and 0 < x < template_width - bubble_w:
                bubble = img[y:y+bubble_h, x:x+bubble_w]
                gray = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)
                darkness = np.mean(gray)

                # Color based on darkness
                if darkness < 100:
                    color = (0, 0, 255)  # Red = filled
                elif darkness < 150:
                    color = (0, 165, 255)  # Orange = possibly filled
                else:
                    color = (0, 255, 0)  # Green = empty

                cv2.rectangle(output, (x, y), (x+bubble_w, y+bubble_h), color, 2)

                # Add letter label on first column
                if col == 0 and row < 10:
                    cv2.putText(output, f"{letter}:{int(darkness)}", (x-60, y+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Crop name area for visualization
    name_area = output[400:1700, 100:600]
    filename = f"verify_{config_name.replace(' ', '_')}.jpg"
    cv2.imwrite(filename, name_area)
    print(f"Saved {filename}")

# Also create a combined analysis
print("\n--- Detailed analysis for HoughCircles config ---")
origin_x, origin_y = 192, 742
bubbles_gap, labels_gap = 51, 54

print("First column bubble darkness values:")
for row in range(26):
    y = origin_y + row * bubbles_gap
    x = origin_x
    letter = chr(65 + row)

    if 0 < y < template_height - bubble_h and 0 < x < template_width - bubble_w:
        bubble = img[y:y+bubble_h, x:x+bubble_w]
        gray = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)
        darkness = np.mean(gray)

        marker = ""
        if darkness < 100:
            marker = " <-- FILLED"
        elif darkness < 150:
            marker = " <-- possibly filled"

        print(f"  Row {letter}: y={y}, darkness={darkness:.1f}{marker}")

# The expected name is "PARI KUMARI"
# So we expect: Col1=P, Col2=A, Col3=R, Col4=I, Col5=space, Col6=K, etc.
print("\n--- Expected vs Detected for 'PARI KUMARI' ---")
expected = "PARI KUMARI"
print(f"Expected name: '{expected}'")

detected_chars = []
for col in range(len(expected)):
    x = origin_x + col * labels_gap
    best_row = None
    best_darkness = 255

    for row in range(26):
        y = origin_y + row * bubbles_gap
        letter = chr(65 + row)

        if 0 < y < template_height - bubble_h and 0 < x < template_width - bubble_w:
            bubble = img[y:y+bubble_h, x:x+bubble_w]
            gray = cv2.cvtColor(bubble, cv2.COLOR_BGR2GRAY)
            darkness = np.mean(gray)

            if darkness < best_darkness:
                best_darkness = darkness
                best_row = letter

    if best_darkness < 150:
        detected_chars.append(best_row if best_row else ' ')
    else:
        detected_chars.append(' ')

detected_name = ''.join(detected_chars)
print(f"Detected name: '{detected_name}'")

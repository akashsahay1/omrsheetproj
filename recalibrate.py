"""Recalibrate bubble positions by finding actual bubble centers."""
import cv2
import numpy as np

img = cv2.imread("filledomrsheet/41_624.jpg")
img = cv2.resize(img, (1360, 1928))

# Look at a wider area to find the actual bubbles
# The name bubbles should be around y=380-1100
name_area = img[370:450, 95:250]
cv2.imwrite("test_first_row_area.jpg", name_area)

# Convert to grayscale and find circles (bubbles)
gray = cv2.cvtColor(name_area, cv2.COLOR_BGR2GRAY)

# Use HoughCircles to detect the bubble positions
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=15,  # Minimum distance between circles
    param1=50,
    param2=25,
    minRadius=5,
    maxRadius=12
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"Found {len(circles[0])} circles")

    # Draw detected circles
    output = name_area.copy()
    for i in circles[0,:]:
        # Draw circle center
        cv2.circle(output, (i[0], i[1]), 2, (0, 255, 0), 2)
        # Draw circle outline
        cv2.circle(output, (i[0], i[1]), i[2], (0, 0, 255), 1)
        print(f"Circle at ({i[0]}, {i[1]}), radius={i[2]}")

    cv2.imwrite("test_detected_circles.jpg", output)
    print("Saved detected circles")

    # Find the first column of bubbles (should be around x=10-25)
    first_col = [c for c in circles[0] if c[0] < 30]
    first_col.sort(key=lambda c: c[1])  # Sort by y position
    print(f"\nFirst column bubbles (sorted by y):")
    for c in first_col[:5]:
        actual_y = 370 + c[1]  # Convert back to full image coords
        actual_x = 95 + c[0]
        print(f"  x={actual_x}, y={actual_y}")
else:
    print("No circles detected")

# Also try template matching approach - find a single bubble and measure spacing
print("\n--- Manual measurement ---")
# Looking at the image, let's find where row A actually starts
# The first bubble row should be around y=387 as per template
# Let's check different y values
for test_y in range(380, 420, 5):
    row = img[test_y:test_y+14, 100:120]
    mean_val = np.mean(cv2.cvtColor(row, cv2.COLOR_BGR2GRAY))
    print(f"y={test_y}: mean brightness = {mean_val:.1f}")

"""
Verify Roll Number coordinates by checking what's at each expected position
"""
import cv2
import numpy as np

def verify():
    img = cv2.imread('omrfilled/omr-filled.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Template coordinates
    origin = (1624, 1278)
    bubbles_gap = 54
    labels_gap = 56

    print("Verifying Roll Number coordinates")
    print(f"Origin: {origin}, BubblesGap: {bubbles_gap}, LabelsGap: {labels_gap}")
    print("Expected: 017")
    print("=" * 60)

    # Check each column
    for col in range(3):
        col_x = origin[0] + col * labels_gap
        print(f"\nColumn {col+1} (x={col_x}):")

        detected = []
        for row in range(10):  # 0-9
            y = origin[1] + row * bubbles_gap
            roi = gray[y-12:y+12, col_x-12:col_x+12]
            if roi.size > 0:
                mean = np.mean(roi)
                if mean < 150:
                    detected.append((row, mean))
                    print(f"  Row {row} (y={y}): mean={mean:.1f} -> {'FILLED' if mean < 100 else 'maybe'}")

        if detected:
            # Find the darkest (most filled)
            darkest = min(detected, key=lambda x: x[1])
            print(f"  => Darkest: Row {darkest[0]} with mean={darkest[1]:.1f}")

    # Also let's check what the actual positions are
    print("\n" + "=" * 60)
    print("Checking actual dark positions found earlier:")
    actual_positions = [
        (1624, 1278, "Expected col1-row0"),
        (1680, 1334, "Expected col2-row1"),
        (1750, 1655, "Expected col3-row7"),
    ]

    for x, y, desc in actual_positions:
        roi = gray[y-12:y+12, x-12:x+12]
        if roi.size > 0:
            mean = np.mean(roi)
            status = "FILLED" if mean < 100 else ("maybe" if mean < 150 else "empty")
            print(f"  {desc} ({x},{y}): mean={mean:.1f} -> {status}")

if __name__ == '__main__':
    verify()

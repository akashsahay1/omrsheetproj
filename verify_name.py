"""
Verify name detection with current coordinates
"""
import cv2
import numpy as np

def verify_name():
    img = cv2.imread('omrfilled/omr-filled.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Current template coordinates
    origin = (142, 477)
    bubbles_gap = 28
    labels_gap = 54
    bubble_size = 28

    # Expected: DEV MANDAL
    # D=3, E=4, V=21, space, M=12, A=0, N=13, D=3, A=0, L=11
    expected = {
        1: ('D', 3), 2: ('E', 4), 3: ('V', 21),
        5: ('M', 12), 6: ('A', 0), 7: ('N', 13),
        8: ('D', 3), 9: ('A', 0), 10: ('L', 11)
    }

    print("=== Verifying Name Detection ===")
    print(f"Origin: {origin}, BubblesGap: {bubbles_gap}, LabelsGap: {labels_gap}")
    print()

    # Check each expected character
    for col in range(1, 11):
        col_x = origin[0] + ((col-1) * labels_gap)
        print(f"Column {col} (x={col_x}):")

        # Check what's detected at each letter position
        detected = []
        for letter_idx in range(26):
            letter = chr(65 + letter_idx)
            y = origin[1] + (letter_idx * bubbles_gap)

            # Extract ROI
            x1, y1 = col_x - bubble_size//2, y - bubble_size//2
            x2, y2 = col_x + bubble_size//2, y + bubble_size//2
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            roi = gray[y1:y2, x1:x2]
            if roi.size > 0:
                mean_intensity = np.mean(roi)
                dark_pixels = np.sum(roi < 180)
                fill_ratio = dark_pixels / roi.size

                if fill_ratio > 0.32:
                    detected.append((letter, fill_ratio, y))

        if detected:
            for letter, fill, y in detected:
                exp = expected.get(col, ('?', -1))
                match = "OK" if letter == exp[0] else "XX"
                print(f"  {match} Detected {letter} (fill={fill:.2f}) at y={y}, expected {exp[0]}")
        else:
            exp = expected.get(col, ('?', -1))
            if col in expected:
                print(f"  XX Nothing detected, expected {exp[0]}")
            else:
                print(f"  - Nothing detected (expected space)")

        print()

    # Now find actual filled bubble positions in each column
    print("\n=== Finding actual filled positions per column ===")
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    for col in range(1, 11):
        col_x = origin[0] + ((col-1) * labels_gap)
        # Search in a narrow vertical strip for this column
        x1, x2 = col_x - 20, col_x + 20

        col_strip = binary[400:1200, max(0,x1):min(w,x2)]
        contours, _ = cv2.findContours(col_strip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filled = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:  # Significant filled area
                x, y, bw, bh = cv2.boundingRect(cnt)
                cy = 400 + y + bh//2
                filled.append((cy, area))

        filled.sort(key=lambda p: p[0])
        print(f"Col {col} (x={col_x}): {len(filled)} filled regions")
        for y, area in filled[:3]:
            # Calculate which letter this corresponds to
            letter_idx = round((y - origin[1]) / bubbles_gap)
            if 0 <= letter_idx < 26:
                letter = chr(65 + letter_idx)
            else:
                letter = "?"
            print(f"    y={y} (area={area:.0f}) -> letter index {letter_idx} = {letter}")

if __name__ == '__main__':
    verify_name()

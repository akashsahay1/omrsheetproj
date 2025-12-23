"""
Search the middle section for Roll No. table
"""
import cv2
import numpy as np

def search():
    img = cv2.imread('omrfilled/omr-filled.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Searching middle section (x=850-1350)")

    # Crop the middle section where Roll No. should be
    crops = [
        ('middle1', 450, 850, 850, 1250),
        ('middle2', 500, 900, 900, 1300),
        ('middle3', 550, 950, 950, 1400),
    ]

    for name, y1, y2, x1, x2 in crops:
        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f'debug_{name}.png', crop)
        print(f"Saved debug_{name}.png ({x1},{y1}) to ({x2},{y2})")

    # Search for dark filled bubbles in this area
    print("\n--- Dark spots in x=850-1400, y=500-950 ---")
    for y in range(520, 920, 25):
        for x in range(880, 1350, 25):
            roi = gray[y-12:y+12, x-12:x+12]
            if roi.size > 0:
                mean = np.mean(roi)
                if mean < 50:
                    print(f"  ({x}, {y}) mean={mean:.1f} FILLED")

if __name__ == '__main__':
    search()

"""
View the entire right section to find Roll No. table
"""
import cv2

img = cv2.imread('omrfilled/omr-filled.png')

# The entire right panel from y=100 to y=1200, x=1200 to edge
sections = [
    ('right_top', 100, 600, 1200, 2400),
    ('right_mid', 500, 1000, 1200, 2400),
    ('right_bot', 900, 1400, 1200, 2400),
]

for name, y1, y2, x1, x2 in sections:
    crop = img[y1:y2, x1:min(x2, img.shape[1])]
    cv2.imwrite(f'debug_{name}.png', crop)
    print(f"Saved debug_{name}.png")

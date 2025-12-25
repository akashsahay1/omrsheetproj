import cv2
import numpy as np

# Read and resize reference image
img = cv2.imread(r'F:\Projects\omrsheetproj\omrfilled\omr-filled.png')
target_w, target_h = 1360, 1928
img = cv2.resize(img, (target_w, target_h))

# Save different regions for examination
regions = [
    ('header', 0, 150, 0, 500),        # y1, y2, x1, x2 - top header
    ('name_area', 100, 400, 20, 520),  # NAME grid area
    ('right_side', 350, 600, 700, 1350),  # Roll/Class/Section area
    ('answers', 600, 950, 50, 850),    # Answer bubbles area
]

for name, y1, y2, x1, x2 in regions:
    region = img[y1:y2, x1:x2]
    path = f'F:\\Projects\\omrsheetproj\\region_{name}.png'
    cv2.imwrite(path, region)
    print(f'Saved {name}: y={y1}-{y2}, x={x1}-{x2}')

print('\nForm layout at 1360x1928:')
print('- Header with logos: y=0-100')
print('- NAME grid label: around y=100-120')
print('- NAME bubbles: starting around y=120')

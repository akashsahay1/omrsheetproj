import cv2
import numpy as np

# Read reference image
ref = cv2.imread(r'F:\Projects\omrsheetproj\omrfilled\omr-filled.png')
h, w = ref.shape[:2]
print(f'Reference: {w}x{h}')

# User-provided marker positions (center of each marker)
markers = {
    'top_left': (141, 560),
    'top_right': (2415, 567),
    'bottom_left': (136, 3146),
    'bottom_right': (2408, 3149)
}

# Extract just the form content (between markers)
# Add small margin to include markers
margin = 20
x1 = min(markers['top_left'][0], markers['bottom_left'][0]) - margin
y1 = min(markers['top_left'][1], markers['top_right'][1]) - margin
x2 = max(markers['top_right'][0], markers['bottom_right'][0]) + margin
y2 = max(markers['bottom_left'][1], markers['bottom_right'][1]) + margin

# Crop to marker-bounded region
cropped = ref[y1:y2, x1:x2]
print(f'Cropped region: {cropped.shape[1]}x{cropped.shape[0]}')

# Resize to OMRChecker processing dimensions
target_w, target_h = 1360, 1928
resized = cv2.resize(cropped, (target_w, target_h))
print(f'Resized to: {target_w}x{target_h}')

# Save for calibration
cv2.imwrite(r'F:\Projects\omrsheetproj\reference_calibration.png', resized)
print('\nSaved reference_calibration.png')

# Calculate scale factors
crop_w = x2 - x1
crop_h = y2 - y1
scale_x = target_w / crop_w
scale_y = target_h / crop_h
print(f'\nScale factors: x={scale_x:.4f}, y={scale_y:.4f}')

# Calculate new marker positions in the resized image
print('\nMarker positions in 1360x1928 image:')
for name, (mx, my) in markers.items():
    new_x = int((mx - x1) * scale_x)
    new_y = int((my - y1) * scale_y)
    print(f'  {name}: ({new_x}, {new_y})')

print('\nPlease use pixspy to measure bubble positions on reference_calibration.png')

import cv2
import numpy as np

# Read reference image
img = cv2.imread(r'F:\Projects\omrsheetproj\omrfilled\omr-filled.png')
h, w = img.shape[:2]
print(f'Reference image: {w}x{h}')

# Corner marker positions (from user measurement)
markers = {
    'top_left': (141, 560),
    'top_right': (2415, 567),
    'bottom_left': (136, 3146),
    'bottom_right': (2408, 3149)
}

# Print marker positions
print('\nCorner markers:')
for name, (x, y) in markers.items():
    print(f'  {name}: ({x}, {y})')

# Extract marker template from top-left (40x40 region centered on marker)
mx, my = markers['top_left']
marker_size = 40
half = marker_size // 2
marker = img[my-half:my+half, mx-half:mx+half]
marker_path = r'F:\Projects\omrsheetproj\OMRChecker\inputs\VidyaBharti2025\omr_marker.jpg'
cv2.imwrite(marker_path, marker)
print(f'\nSaved marker template ({marker_size}x{marker_size}) to omr_marker.jpg')

# Verify marker looks correct
gray_marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
print(f'Marker pixel range: {gray_marker.min()} - {gray_marker.max()}')
print(f'Marker mean: {gray_marker.mean():.1f}')

# Calculate the reference dimensions for CropOnMarkers
# The 4 markers define the corners of the form area
# CropOnMarkers will warp images to fit within these points
form_width = markers['top_right'][0] - markers['top_left'][0]
form_height = markers['bottom_left'][1] - markers['top_left'][1]
print(f'\nForm dimensions (between markers): {form_width}x{form_height}')

# Scale to 1360x1928 processing size
scale_x = 1360 / w
scale_y = 1928 / h
print(f'\nScale factors to 1360x1928: x={scale_x:.4f}, y={scale_y:.4f}')

# Scaled marker positions
print('\nScaled marker positions (for 1360x1928):')
for name, (x, y) in markers.items():
    sx = int(x * scale_x)
    sy = int(y * scale_y)
    print(f'  {name}: ({sx}, {sy})')

import cv2
import numpy as np

# Read reference image
img = cv2.imread(r'F:\Projects\omrsheetproj\omrfilled\omr-filled.png')
print(f'Image shape: {img.shape}')

# The top-left corner marker is visible at approximately (50, 100)
# Extract top-left corner marker (black circle)
marker = img[75:125, 45:95]  # y1:y2, x1:x2
cv2.imwrite(r'F:\Projects\omrsheetproj\OMRChecker\inputs\VidyaBharti2025\omr_marker.jpg', marker)
print('Saved marker to omr_marker.jpg')
print(f'Marker shape: {marker.shape}')

# Check pixel values
gray = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)
print(f'Min pixel: {gray.min()}, Max: {gray.max()}, Mean: {gray.mean():.1f}')

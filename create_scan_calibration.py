import cv2

# Read a scanned image
scan = cv2.imread(r'F:\Projects\omrsheetproj\filledomrsheet\41_624.jpg')
h, w = scan.shape[:2]
print(f'Original scan: {w}x{h}')

# Resize to OMRChecker processing dimensions (simple resize, no perspective)
target_w, target_h = 1360, 1928
resized = cv2.resize(scan, (target_w, target_h))
print(f'Resized to: {target_w}x{target_h}')

# Save for calibration
cv2.imwrite(r'F:\Projects\omrsheetproj\scan_calibration.png', resized)
print('\nSaved scan_calibration.png')
print('Measure bubble positions on this image with pixspy.')

import cv2
import os

# The OMRChecker already processed the image correctly
# Let's extract the aligned image from its internal processing

# Read the CheckedOMR output - this shows the aligned image with annotations
checked = cv2.imread(r'F:\Projects\omrsheetproj\outputs\_test_crop\CheckedOMRs\41_624.jpg')
print(f'Checked image: {checked.shape[1]}x{checked.shape[0]}')

# This is already the warped/aligned image at processing dimensions
# Save it for calibration
cv2.imwrite(r'F:\Projects\omrsheetproj\aligned_for_calibration.png', checked)
print('Saved aligned_for_calibration.png')

# Also check what dimensions OMRChecker used
print(f'\nThis image is {checked.shape[1]}x{checked.shape[0]} pixels')
print('Measure bubble positions on this image.')

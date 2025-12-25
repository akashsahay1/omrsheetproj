import sys
import os
import shutil

# Create test input directory
test_dir = r'F:\Projects\omrsheetproj\OMRChecker\inputs\_test_crop'
os.makedirs(test_dir, exist_ok=True)

# Copy template and marker files
shutil.copy(r'F:\Projects\omrsheetproj\OMRChecker\inputs\VidyaBharti2025\template.json', test_dir)
shutil.copy(r'F:\Projects\omrsheetproj\OMRChecker\inputs\VidyaBharti2025\omr_marker.jpg', test_dir)

# Copy one test image
test_image = r'F:\Projects\omrsheetproj\filledomrsheet\41_624.jpg'
shutil.copy(test_image, test_dir)

print(f'Test directory ready: {test_dir}')
print(f'Contents: {os.listdir(test_dir)}')
print('\nRun OMRChecker with:')
print(f'cd F:\\Projects\\omrsheetproj\\OMRChecker && ..\\venv\\Scripts\\python.exe main.py -i inputs\\_test_crop -o ..\\outputs\\_test_crop')

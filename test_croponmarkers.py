import sys
import os
import shutil

# Add OMRChecker to path
sys.path.insert(0, r'F:\Projects\omrsheetproj\OMRChecker')

# Create a test directory with one image
test_dir = r'F:\Projects\omrsheetproj\OMRChecker\inputs\_test_crop'
os.makedirs(test_dir, exist_ok=True)

# Copy template and marker files
shutil.copy(r'F:\Projects\omrsheetproj\OMRChecker\inputs\VidyaBharti2025\template.json', test_dir)
shutil.copy(r'F:\Projects\omrsheetproj\OMRChecker\inputs\VidyaBharti2025\omr_marker.jpg', test_dir)

# Copy one test image
test_image = r'F:\Projects\omrsheetproj\filledomrsheet\41_624.jpg'
shutil.copy(test_image, test_dir)

print(f'Test directory: {test_dir}')
print(f'Contents: {os.listdir(test_dir)}')

# Run OMRChecker
from src.entry import entry_point

output_dir = r'F:\Projects\omrsheetproj\outputs\_test_crop'
os.makedirs(output_dir, exist_ok=True)

try:
    args = type('Args', (), {
        'input_paths': [test_dir],
        'output_dir': output_dir,
        'autoAlign': False
    })()

    entry_point(args)
    print('\nProcessing complete!')

    # Check results
    results_file = os.path.join(output_dir, '_test_crop', 'Results', 'Results.csv')
    if os.path.exists(results_file):
        with open(results_file) as f:
            print('\nResults:')
            print(f.read()[:2000])
    else:
        print(f'Results file not found at {results_file}')

except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()

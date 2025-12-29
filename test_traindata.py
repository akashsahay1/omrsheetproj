"""Test OMR detection on random traindata files."""
import os
import sys
import random

# Check traindata folder - has subfolders Chayan and Khoj
traindata_base = r'F:\Projects\omrsheetprojm\traindata'
print(f"Checking path: {traindata_base}")

# Collect all image files from subfolders
all_files = []
for subfolder in ['Chayan', 'Khoj']:
    folder_path = os.path.join(traindata_base, subfolder)
    if os.path.exists(folder_path):
        folder_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in folder_files:
            all_files.append((subfolder, os.path.join(folder_path, f), f))
        print(f"  {subfolder}: {len(folder_files)} files")

print(f"Total image files: {len(all_files)}")

if len(all_files) == 0:
    print("No image files found!")
    sys.exit(1)

# Select random files (default 30, or pass as command line arg)
default_size = 50
sample_size = min(int(sys.argv[1]) if len(sys.argv) > 1 else default_size, len(all_files))
random_files = random.sample(all_files, sample_size)
print(f"Selected {sample_size} random files for testing\n")

# Import the processor
sys.path.insert(0, r'F:\Projects\omrsheetprojm\omrsheetproj')
from src.ai.yolo_omr_processor import YOLOOMRProcessor

# Initialize processor
processor = YOLOOMRProcessor()

# Print header
print(f"{'#':<3} {'Batch':<7} {'Filename':<15} {'Name':<28} {'Roll':<6} {'Class':<6} {'Sec':<4} {'School':<8} {'Conf':<5}")
print("-" * 105)

# Process each file
for i, (batch, filepath, filename) in enumerate(sorted(random_files, key=lambda x: x[2]), 1):
    result = processor.process(filepath)

    # Truncate name if too long
    name = result.name[:25] + "..." if len(result.name) > 28 else result.name

    # Truncate filename if too long
    fname = filename[:12] + "..." if len(filename) > 15 else filename

    print(f"{i:<3} {batch:<7} {fname:<15} {name:<28} {result.roll_number:<6} {result.class_:<6} {result.section:<4} {result.school_code:<8} {result.confidence:<5.0f}")

print("-" * 105)
print("Done!")

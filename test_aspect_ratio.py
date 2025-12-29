"""Test aspect ratio variation across images and propose dynamic scaling."""
import cv2
import os
import sys
import random

sys.path.insert(0, r'F:\Projects\omrsheetprojm\omrsheetproj')

# Target dimensions
TARGET_W = 2480
TARGET_H = 3507
TARGET_ASPECT = TARGET_W / TARGET_H

# Current fixed bubble config
FIXED_BUBBLE_SIZE = (33, 33)
FIXED_ORIGIN_KHOJ = (189, 671)
FIXED_ROWS_GAP = 52
FIXED_COLS_GAP = 54

traindata = r'F:\Projects\omrsheetprojm\traindata'

# Get sample files
khoj_files = [os.path.join(traindata, 'Khoj', f)
              for f in os.listdir(os.path.join(traindata, 'Khoj'))
              if f.endswith('.jpg')][:5]
chayan_files = [os.path.join(traindata, 'Chayan', f)
                for f in os.listdir(os.path.join(traindata, 'Chayan'))
                if f.endswith('.jpg')][:5]

print('Image Dimension Analysis:')
print('=' * 90)
print(f"{'File':<20} {'Width':>7} {'Height':>7} {'Aspect':>8} {'W_Scale':>8} {'H_Scale':>8} {'Distort%':>9}")
print('-' * 90)

all_files = khoj_files + chayan_files

for f in all_files:
    img = cv2.imread(f)
    if img is not None:
        h, w = img.shape[:2]
        aspect = w / h
        w_scale = TARGET_W / w
        h_scale = TARGET_H / h
        distortion = abs(w_scale - h_scale) / min(w_scale, h_scale) * 100
        fname = os.path.basename(f)
        print(f"{fname:<20} {w:>7} {h:>7} {aspect:>8.4f} {w_scale:>8.4f} {h_scale:>8.4f} {distortion:>8.2f}%")

print('-' * 90)
print(f"Target: {TARGET_W} x {TARGET_H}, Aspect: {TARGET_ASPECT:.4f}")
print()

# Now test with dynamic scaling
print("\n" + "=" * 90)
print("Testing DYNAMIC scaling vs FIXED scaling on name detection:")
print("=" * 90)

from src.ai.yolo_omr_processor import YOLOOMRProcessor
import numpy as np

def detect_with_dynamic_scaling(image_path):
    """Detect name using dynamically scaled bubble parameters."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Calculate scale factors
    w_scale = TARGET_W / w
    h_scale = TARGET_H / h

    # Resize image (same as current)
    aligned = cv2.resize(img, (TARGET_W, TARGET_H))
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    # Dynamic bubble parameters based on AVERAGE scale
    # Since we're resizing, we need to scale the bubble detection area
    # If image was stretched more horizontally, bubbles appear wider
    avg_scale = (w_scale + h_scale) / 2
    scale_ratio = w_scale / h_scale  # >1 means stretched horizontally

    # Adjust bubble size based on how much distortion occurred
    # If w_scale > h_scale, bubbles are stretched horizontally
    bubble_w = int(FIXED_BUBBLE_SIZE[0] * (h_scale / w_scale)) if w_scale > h_scale else FIXED_BUBBLE_SIZE[0]
    bubble_h = int(FIXED_BUBBLE_SIZE[1] * (w_scale / h_scale)) if h_scale > w_scale else FIXED_BUBBLE_SIZE[1]

    # Keep bubble size reasonable
    bubble_w = max(28, min(40, bubble_w))
    bubble_h = max(28, min(40, bubble_h))

    return (w, h, w_scale, h_scale, bubble_w, bubble_h)

# Test on a few files
print(f"\n{'File':<20} {'Orig WxH':>15} {'W_Scale':>8} {'H_Scale':>8} {'Adj Bubble':>12}")
print('-' * 70)

for f in all_files[:5]:
    result = detect_with_dynamic_scaling(f)
    w, h, w_scale, h_scale, bw, bh = result
    fname = os.path.basename(f)
    print(f"{fname:<20} {w:>6}x{h:<7} {w_scale:>8.4f} {h_scale:>8.4f} {bw:>5}x{bh:<5}")

print()
print("Note: If W_Scale ≠ H_Scale, the image aspect ratio differs from target.")
print("      Current fixed 33x33 bubble may not capture the actual bubble correctly.")

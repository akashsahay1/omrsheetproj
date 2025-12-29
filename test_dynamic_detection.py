"""Compare detection with fixed vs dynamic bubble scaling."""
import cv2
import os
import sys
import numpy as np

sys.path.insert(0, r'F:\Projects\omrsheetprojm\omrsheetproj')

# Target dimensions
TARGET_W = 2480
TARGET_H = 3507

# Fixed config (current)
FIXED_ORIGIN = (189, 671)
FIXED_BUBBLE = (33, 33)
FIXED_ROWS_GAP = 52
FIXED_COLS_GAP = 54

traindata = r'F:\Projects\omrsheetprojm\traindata'

def detect_name_fixed(gray, y_offset=0):
    """Detect name using FIXED bubble parameters."""
    origin = FIXED_ORIGIN
    bubble_size = FIXED_BUBBLE
    rows_gap = FIXED_ROWS_GAP
    cols_gap = FIXED_COLS_GAP
    values = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    result = []
    for col in range(20):  # First 20 columns
        x = origin[0] + col * cols_gap
        bubble_data = []
        for row in range(26):
            y = origin[1] + y_offset + row * rows_gap
            bw, bh = bubble_size
            if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                bubble = gray[y:y+bh, x:x+bw]
                darkness = np.mean(bubble)
                bubble_data.append((row, darkness))

        if bubble_data:
            sorted_b = sorted(bubble_data, key=lambda x: x[1])
            darkest_row, darkest_val = sorted_b[0]
            median = np.median([v for _, v in bubble_data])
            gap = median - darkest_val
            if darkest_val < 180 and gap > 30:
                result.append(values[darkest_row])
            elif darkest_val < 195 and gap > 40:
                result.append(values[darkest_row])
            else:
                result.append(' ')
    return ''.join(result).strip()


def detect_name_dynamic(image_path, y_offset=0):
    """Detect name using DYNAMIC bubble parameters based on aspect ratio."""
    img = cv2.imread(image_path)
    orig_h, orig_w = img.shape[:2]

    # Calculate scale factors
    w_scale = TARGET_W / orig_w
    h_scale = TARGET_H / orig_h

    # Resize
    aligned = cv2.resize(img, (TARGET_W, TARGET_H))
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    # Adjust parameters based on scale ratio
    # The bubble coordinates were calibrated assuming uniform scaling
    # If h_scale > w_scale, vertical stretch occurred, so reduce vertical params
    scale_ratio = h_scale / w_scale

    # Adjust bubble size
    bubble_w = FIXED_BUBBLE[0]
    bubble_h = int(FIXED_BUBBLE[1] / scale_ratio)  # Compensate for vertical stretch

    # Adjust gaps
    rows_gap = int(FIXED_ROWS_GAP / scale_ratio)
    cols_gap = FIXED_COLS_GAP  # Horizontal gap less affected

    # Adjust origin Y
    origin_x = FIXED_ORIGIN[0]
    origin_y = int(FIXED_ORIGIN[1] / scale_ratio)

    values = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    result = []

    for col in range(20):
        x = origin_x + col * cols_gap
        bubble_data = []
        for row in range(26):
            y = origin_y + y_offset + row * rows_gap
            if 0 <= y < gray.shape[0] - bubble_h and 0 <= x < gray.shape[1] - bubble_w:
                bubble = gray[y:y+bubble_h, x:x+bubble_w]
                darkness = np.mean(bubble)
                bubble_data.append((row, darkness))

        if bubble_data:
            sorted_b = sorted(bubble_data, key=lambda x: x[1])
            darkest_row, darkest_val = sorted_b[0]
            median = np.median([v for _, v in bubble_data])
            gap = median - darkest_val
            if darkest_val < 180 and gap > 30:
                result.append(values[darkest_row])
            elif darkest_val < 195 and gap > 40:
                result.append(values[darkest_row])
            else:
                result.append(' ')

    return ''.join(result).strip(), (bubble_w, bubble_h), rows_gap, scale_ratio


# Test on specific file and some random files
test_files = [
    os.path.join(traindata, 'Khoj', '10447A.jpg'),  # The problematic file
    os.path.join(traindata, 'Khoj', '10001A.jpg'),
    os.path.join(traindata, 'Khoj', '10253A.jpg'),
    os.path.join(traindata, 'Chayan', '39_1.jpg'),
    os.path.join(traindata, 'Chayan', '39_100.jpg'),
]

print("Comparing FIXED vs DYNAMIC bubble detection:")
print("=" * 100)
print(f"{'File':<18} {'Fixed Bubble':<12} {'Dynamic Bubble':<14} {'D_RowGap':<9} {'Ratio':<7} {'Match?':<6}")
print("-" * 100)

for f in test_files:
    if not os.path.exists(f):
        continue

    # Load and resize for fixed detection
    img = cv2.imread(f)
    aligned = cv2.resize(img, (TARGET_W, TARGET_H))
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    fixed_name = detect_name_fixed(gray, y_offset=5)
    dynamic_name, d_bubble, d_row_gap, ratio = detect_name_dynamic(f, y_offset=5)

    match = "YES" if fixed_name == dynamic_name else "NO"
    fname = os.path.basename(f)

    print(f"{fname:<18} 33x33        {d_bubble[0]}x{d_bubble[1]:<11} {d_row_gap:<9} {ratio:<7.4f} {match:<6}")
    print(f"    Fixed:   [{fixed_name}]")
    print(f"    Dynamic: [{dynamic_name}]")
    print()

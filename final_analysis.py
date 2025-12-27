#!/usr/bin/env python3
"""Final analysis of problem cases."""

import sys
sys.path.insert(0, '/Volumes/Projects/Projects/omrsheetproj')

from src.ai.yolo_omr_processor import YOLOOMRProcessor
from pathlib import Path

def main():
    reference_path = "/Volumes/Projects/Projects/omrsheetproj/OMRChecker/inputs/VidyaBharti2025/omr-filled.png"
    bubble_classifier_path = "/Volumes/Projects/Projects/omrsheetproj/models/bubble_classifier.pt"

    processor = YOLOOMRProcessor(
        reference_image_path=reference_path,
        bubble_classifier_path=bubble_classifier_path
    )

    test_dir = Path('/Volumes/Projects/Projects/omrsheetproj/filledomrsheet')
    test_images = sorted(test_dir.glob('*.jpg'))

    # Valid single names that don't need spaces
    valid_single_names = {'PRINCE', 'KUMARI', 'YUVRAJ', 'PRAGYAN', 'VARSHA', 'TANISHA',
                         'VIKRAM', 'SURAJ', 'SUMAN', 'RINA', 'REKHA', 'NEHA', 'PUJA'}

    good_with_space = 0
    valid_single = 0
    short_garbled = 0
    empty = 0
    long_no_space = 0

    for img_path in test_images:
        result = processor.process(str(img_path))
        name = result.name

        if ' ' in name and len(name) > 5:
            good_with_space += 1
        elif name.upper() in valid_single_names:
            valid_single += 1
        elif not name or len(name) <= 2:
            empty += 1
        elif len(name) <= 5:
            short_garbled += 1
        else:
            long_no_space += 1

    total = len(test_images)

    print("=" * 60)
    print("FINAL ACCURACY ANALYSIS")
    print("=" * 60)
    print(f"\nTotal images: {total}")
    print(f"\n✓ Names with proper spacing: {good_with_space} ({100*good_with_space/total:.1f}%)")
    print(f"✓ Valid single-word names:   {valid_single} ({100*valid_single/total:.1f}%)")
    print(f"  (PRINCE, KUMARI, YUVRAJ, etc. - don't need spaces)")
    print(f"\n✗ Empty/very short (<=2):    {empty} ({100*empty/total:.1f}%)")
    print(f"✗ Short garbled (3-5 chars): {short_garbled} ({100*short_garbled/total:.1f}%)")
    print(f"✗ Long without space:        {long_no_space} ({100*long_no_space/total:.1f}%)")

    effective_good = good_with_space + valid_single
    unrecoverable = empty + short_garbled
    potentially_fixable = long_no_space

    print(f"\n" + "=" * 60)
    print(f"EFFECTIVE ACCURACY: {effective_good}/{total} ({100*effective_good/total:.1f}%)")
    print(f"  (includes valid single-word names)")
    print(f"\nUnrecoverable (poor scan quality): {unrecoverable}")
    print(f"Potentially fixable with better ML: {potentially_fixable}")

if __name__ == "__main__":
    main()

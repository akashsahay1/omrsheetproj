#!/usr/bin/env python3
"""Test YOLO OMR processor with ML bubble classifier."""

import sys
sys.path.insert(0, '/Volumes/Projects/Projects/omrsheetproj')

from src.ai.yolo_omr_processor import YOLOOMRProcessor
from pathlib import Path

def main():
    # Reference image for alignment
    reference_path = "/Volumes/Projects/Projects/omrsheetproj/OMRChecker/inputs/VidyaBharti2025/omr-filled.png"
    bubble_classifier_path = "/Volumes/Projects/Projects/omrsheetproj/models/bubble_classifier.pt"

    # Initialize processor with ML classifier
    processor = YOLOOMRProcessor(
        reference_image_path=reference_path,
        bubble_classifier_path=bubble_classifier_path
    )

    # Test on sample images
    test_dir = Path('/Volumes/Projects/Projects/omrsheetproj/filledomrsheet')
    test_images = sorted(test_dir.glob('*.jpg'))[:20]

    print(f"Testing YOLO OMR Processor with ML classifier on {len(test_images)} images")
    print("=" * 80)
    print(f"{'Filename':<20} {'Name':<35} {'Roll':<8} {'Class':<6} {'Section'}")
    print("-" * 80)

    good = 0
    for img_path in test_images:
        result = processor.process(str(img_path))
        name = result.name if result.name else "(empty)"
        name_display = name[:32] + "..." if len(name) > 35 else name

        # Check if name has proper spacing
        has_space = ' ' in result.name and len(result.name) > 5
        status = "OK" if has_space else "  "
        if has_space:
            good += 1

        print(f"{img_path.name:<20} {name_display:<35} {result.roll_number:<8} "
              f"{result.class_:<6} {result.section} {status}")

    print("=" * 80)
    print(f"Good names (with space): {good}/{len(test_images)} ({100*good/len(test_images):.1f}%)")

if __name__ == "__main__":
    main()

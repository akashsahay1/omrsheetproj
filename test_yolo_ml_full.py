#!/usr/bin/env python3
"""Test YOLO OMR processor with ML classifier on all images."""

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

    print(f"Testing YOLO OMR Processor with ML classifier on {len(test_images)} images")
    print("=" * 80)

    good = 0
    problem_names = []

    for img_path in test_images:
        result = processor.process(str(img_path))
        name = result.name if result.name else "(empty)"

        has_space = ' ' in result.name and len(result.name) > 5
        if has_space:
            good += 1
        else:
            problem_names.append((img_path.name, name))

    print(f"Names with proper spacing: {good}/{len(test_images)} ({100*good/len(test_images):.1f}%)")
    print(f"Target: 295+ (98%+)")
    print()

    print("Sample problem names (first 30):")
    for filename, name in problem_names[:30]:
        print(f"  {filename:<20} -> {name}")

if __name__ == "__main__":
    main()

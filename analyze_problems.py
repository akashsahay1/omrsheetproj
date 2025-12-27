#!/usr/bin/env python3
"""Analyze problem cases to categorize issues."""

import sys
sys.path.insert(0, '/Volumes/Projects/Projects/omrsheetproj')

from src.ai.yolo_omr_processor import YOLOOMRProcessor
from pathlib import Path

def main():
    reference_path = "/Volumes/Projects/Projects/omrsheetproj/OMRChecker/inputs/VidyaBharti2025/omr-filled.png"
    processor = YOLOOMRProcessor(reference_image_path=reference_path)

    test_dir = Path('/Volumes/Projects/Projects/omrsheetproj/filledomrsheet')
    test_images = sorted(test_dir.glob('*.jpg'))

    # Categorize results
    good = []  # Has space, looks correct
    short_names = []  # Less than 5 chars total (likely missing letters)
    no_space_long = []  # No space but >= 8 chars (surname not found)
    single_word_names = []  # Could be valid single names (KUMARI, PRINCE)
    empty = []  # Empty or very short (1-2 chars)

    for img_path in test_images:
        result = processor.process(str(img_path))
        name = result.name

        has_space = ' ' in name and len(name) > 5

        if has_space:
            good.append((img_path.name, name))
        elif not name or len(name) <= 2:
            empty.append((img_path.name, name))
        elif len(name) <= 5:
            short_names.append((img_path.name, name))
        elif len(name) >= 8 and ' ' not in name:
            no_space_long.append((img_path.name, name))
        else:
            single_word_names.append((img_path.name, name))

    print("=" * 80)
    print(f"ANALYSIS OF {len(test_images)} IMAGES")
    print("=" * 80)
    print(f"\n✓ Good (has space): {len(good)}")
    print(f"\n✗ Empty/very short (<=2 chars): {len(empty)}")
    for f, n in empty[:10]:
        print(f"    {f}: '{n}'")

    print(f"\n✗ Short names (3-5 chars, likely missing letters): {len(short_names)}")
    for f, n in short_names[:15]:
        print(f"    {f}: '{n}'")

    print(f"\n✗ Long without space (>=8 chars, surname not found): {len(no_space_long)}")
    for f, n in no_space_long[:15]:
        print(f"    {f}: '{n}'")

    print(f"\n? Single word (6-7 chars, might be valid): {len(single_word_names)}")
    for f, n in single_word_names[:15]:
        print(f"    {f}: '{n}'")

    print("\n" + "=" * 80)
    total_problem = len(empty) + len(short_names) + len(no_space_long) + len(single_word_names)
    print(f"Total: {len(good)} good, {total_problem} problems")
    print(f"  - Empty/very short (unrecoverable): {len(empty)}")
    print(f"  - Short names (mostly unrecoverable): {len(short_names)}")
    print(f"  - Long without space (potentially fixable): {len(no_space_long)}")
    print(f"  - Single word (might be valid): {len(single_word_names)}")

if __name__ == "__main__":
    main()

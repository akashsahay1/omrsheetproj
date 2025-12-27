#!/usr/bin/env python3
"""Test YOLO OMR processor."""

import sys
from pathlib import Path

# Add project root to path (cross-platform)
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from src.ai.yolo_omr_processor import YOLOOMRProcessor

def main():
    # Reference image for alignment (cross-platform path)
    reference_path = project_dir / "OMRChecker" / "inputs" / "VidyaBharti2025" / "omr-filled.png"

    # Initialize processor
    processor = YOLOOMRProcessor(reference_image_path=str(reference_path))

    # Test on sample images (cross-platform path)
    test_dir = project_dir / "filledomrsheet"
    test_images = sorted(test_dir.glob('*.jpg'))[:20]
    
    print(f"Testing YOLO OMR Processor on {len(test_images)} images")
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

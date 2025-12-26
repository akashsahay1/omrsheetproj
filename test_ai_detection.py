"""Test AI-based OMR detection."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ai import OMRAIEngine
from pathlib import Path


def test_ai_engine():
    """Test the AI OMR engine on sample images."""
    print("Testing AI-based OMR Detection")
    print("=" * 60)

    # Initialize engine
    reference_image = "OMRChecker/inputs/VidyaBharti2025/omr-filled.png"

    engine = OMRAIEngine(
        reference_image_path=reference_image,
        ocr_backend="easyocr",
        use_gpu=False
    )

    # Test on sample images
    test_images = [
        "filledomrsheet/41_624.jpg",
        "filledomrsheet/41_625.jpg",
        "filledomrsheet/41_630.jpg",
        "filledomrsheet/41_635.jpg",
        "filledomrsheet/41_636.jpg",
    ]

    print(f"\nProcessing {len(test_images)} images...\n")
    print(f"{'Filename':<20} {'Name':<30} {'Roll':<10} {'Class':<8} {'Section':<10}")
    print("-" * 80)

    for img_path in test_images:
        if Path(img_path).exists():
            result = engine.process(img_path)

            if result.success:
                print(f"{Path(img_path).name:<20} {result.name:<30} "
                      f"{result.roll_number:<10} {result.class_:<8} {result.section:<10}")
            else:
                print(f"{Path(img_path).name:<20} ERROR: {result.error_message[:40]}")
        else:
            print(f"{img_path:<20} FILE NOT FOUND")

    print("\n" + "=" * 60)
    print("AI Detection test completed.")


if __name__ == "__main__":
    test_ai_engine()

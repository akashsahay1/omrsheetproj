"""Test Chayan name detection with updated calibration."""

import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai.marker_aligner import MarkerAligner
from ai.field_extractor import FieldExtractor

def main():
    # Test image with known name
    img_path = r"F:\Projects\omrsheetprojm\traindata\mixedbtchs\39_18.jpg"
    expected_name = "KESHAV KUMAR MANDAL"

    print(f"Testing: {img_path}")
    print(f"Expected: {expected_name}")
    print("=" * 50)

    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load image")
        return

    # Align image
    aligner = MarkerAligner()
    result = aligner.align(img)

    if not result.success:
        print("Alignment failed")
        return

    # Extract name
    gray = cv2.cvtColor(result.aligned_image, cv2.COLOR_BGR2GRAY)
    extractor = FieldExtractor()
    name_result = extractor.extract_name(gray, batch_type='chayan')

    print(f"Detected: {name_result.value}")
    print(f"Confidence: {name_result.confidence:.2f}")

    # Compare
    expected_clean = expected_name.replace(" ", "")
    detected_clean = name_result.value.replace(" ", "")

    correct = 0
    for i, (e, d) in enumerate(zip(expected_clean, detected_clean)):
        if e == d:
            correct += 1
        else:
            print(f"  Position {i+1}: expected '{e}', got '{d}'")

    accuracy = correct / len(expected_clean) * 100 if expected_clean else 0
    print(f"\nAccuracy: {correct}/{len(expected_clean)} = {accuracy:.1f}%")


if __name__ == '__main__':
    main()

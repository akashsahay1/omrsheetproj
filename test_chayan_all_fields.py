"""Test all Chayan field extractions with calibrated values."""

import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai.marker_aligner import MarkerAligner
from ai.field_extractor import FieldExtractor


def main():
    # Test image with known values
    img_path = r"F:\Projects\omrsheetprojm\traindata\mixedbtchs\39_18.jpg"

    # Expected values for this image
    expected = {
        'name': 'KESHAV KUMAR MANDAL',
        'roll_number': '030',
        'class': '6',
        'section': 'A',
        'school_code': '025',
    }

    print(f"Testing: {os.path.basename(img_path)}")
    print("=" * 60)

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

    print(f"Image aligned: {result.aligned_image.shape[1]} x {result.aligned_image.shape[0]}")

    # Extract all fields
    gray = cv2.cvtColor(result.aligned_image, cv2.COLOR_BGR2GRAY)
    extractor = FieldExtractor()
    fields = extractor.extract_all_fields(gray, result.aligned_image)

    print("\n" + "=" * 60)
    print("FIELD EXTRACTION RESULTS")
    print("=" * 60)

    # Check each field
    results = []

    # Name
    name_match = fields.name == expected['name']
    results.append(('Name', expected['name'], fields.name, name_match, fields.name_confidence))

    # Roll Number
    roll_match = fields.roll_number == expected['roll_number']
    results.append(('Roll Number', expected['roll_number'], fields.roll_number, roll_match, fields.roll_confidence))

    # Class
    class_match = fields.class_ == expected['class']
    results.append(('Class', expected['class'], fields.class_, class_match, fields.class_confidence))

    # Section
    section_match = fields.section == expected['section']
    results.append(('Section', expected['section'], fields.section, section_match, fields.section_confidence))

    # School Code
    school_match = fields.school_code == expected['school_code']
    results.append(('School Code', expected['school_code'], fields.school_code, school_match, fields.school_code_confidence))

    # Print results table
    print(f"\n{'Field':<15} {'Expected':<25} {'Detected':<25} {'Match':<8} {'Conf':<8}")
    print("-" * 80)

    for field_name, exp_val, det_val, match, conf in results:
        status = "PASS" if match else "FAIL"
        print(f"{field_name:<15} {exp_val:<25} {det_val:<25} {status:<8} {conf:.2f}")

    # Summary
    matches = sum(1 for r in results if r[3])
    total = len(results)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {matches}/{total} fields correct ({matches/total*100:.1f}%)")
    print(f"Overall confidence: {fields.overall_confidence:.1f}%")

    # Print answers detected
    answered = sum(1 for ans in fields.answers.values() if ans)
    print(f"\nAnswers detected: {answered}/60")
    print(f"Answers confidence: {fields.answers_confidence:.1f}%")

    # Show first 20 answers as sample
    print("\nSample answers (Q1-20):")
    for q in range(1, 21):
        ans = fields.answers.get(q, '')
        print(f"  Q{q}: {ans if ans else '-'}", end='')
        if q % 10 == 0:
            print()


if __name__ == '__main__':
    main()

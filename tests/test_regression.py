"""
Regression Tests for OMR Detection

This test suite validates detection accuracy against the golden dataset.
It ensures that bug fixes don't break previously working detections.

Run with: pytest tests/test_regression.py -v
"""

import sys
import json
import pytest
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

GOLDEN_DIR = project_root / "golden_dataset"
IMAGES_DIR = GOLDEN_DIR / "images"
GROUND_TRUTH_FILE = GOLDEN_DIR / "ground_truth.json"


@dataclass
class AccuracyResult:
    """Stores accuracy metrics for a test run."""
    total: int = 0
    correct: int = 0
    errors: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def load_ground_truth() -> Optional[Dict]:
    """Load the golden dataset ground truth file."""
    if not GROUND_TRUTH_FILE.exists():
        return None
    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_verified_images(data: Dict) -> List[Dict]:
    """Get only verified images from ground truth."""
    return [img for img in data.get("images", []) if img.get("verified", False)]


class TestRegressionSuite:
    """Regression test suite for OMR detection accuracy."""

    @pytest.fixture(scope="class")
    def processor(self):
        """Initialize the OMR processor once per test class."""
        from src.ai.yolo_omr_processor import YOLOOMRProcessor
        return YOLOOMRProcessor()

    @pytest.fixture(scope="class")
    def ground_truth(self):
        """Load ground truth data."""
        data = load_ground_truth()
        if data is None:
            pytest.skip("No ground truth file found. Run scripts/create_ground_truth.py first.")
        return data

    @pytest.fixture(scope="class")
    def verified_images(self, ground_truth):
        """Get verified images from ground truth."""
        images = get_verified_images(ground_truth)
        if not images:
            pytest.skip("No verified images in ground truth. Run scripts/create_ground_truth.py first.")
        return images

    def test_ground_truth_exists(self, ground_truth):
        """Verify that ground truth file exists and is valid."""
        assert ground_truth is not None, "Ground truth file not found"
        assert "images" in ground_truth, "Ground truth missing 'images' key"
        assert "accuracy_thresholds" in ground_truth, "Ground truth missing thresholds"

    def test_images_exist(self, verified_images):
        """Verify that all golden dataset images exist."""
        missing = []
        for img_data in verified_images:
            image_path = IMAGES_DIR / img_data["filename"]
            if not image_path.exists():
                missing.append(img_data["filename"])

        assert not missing, f"Missing images: {missing}"

    def test_name_detection_accuracy(self, processor, ground_truth, verified_images):
        """Test name detection accuracy against threshold."""
        threshold = ground_truth["accuracy_thresholds"]["name"]
        result = AccuracyResult()

        for img_data in verified_images:
            expected_name = img_data.get("name", "")
            if not expected_name:  # Skip if no name in ground truth
                continue

            image_path = IMAGES_DIR / img_data["filename"]
            if not image_path.exists():
                continue

            omr_result = processor.process(str(image_path))
            detected_name = omr_result.name if omr_result else ""

            result.total += 1
            if self._names_match(detected_name, expected_name):
                result.correct += 1
            else:
                result.errors.append({
                    "file": img_data["filename"],
                    "expected": expected_name,
                    "detected": detected_name
                })

        # Report accuracy
        print(f"\nName Detection: {result.correct}/{result.total} = {result.accuracy:.1%}")
        if result.errors:
            print("Errors:")
            for err in result.errors[:5]:  # Show first 5 errors
                print(f"  {err['file']}: expected='{err['expected']}' got='{err['detected']}'")

        assert result.accuracy >= threshold, \
            f"Name accuracy {result.accuracy:.1%} below threshold {threshold:.0%}"

    def test_roll_number_detection_accuracy(self, processor, ground_truth, verified_images):
        """Test roll number detection accuracy against threshold."""
        threshold = ground_truth["accuracy_thresholds"]["roll_number"]
        result = AccuracyResult()

        for img_data in verified_images:
            expected_roll = img_data.get("roll_number", "")
            if not expected_roll:
                continue

            image_path = IMAGES_DIR / img_data["filename"]
            if not image_path.exists():
                continue

            omr_result = processor.process(str(image_path))
            detected_roll = omr_result.roll_number if omr_result else ""

            result.total += 1
            if detected_roll == expected_roll:
                result.correct += 1
            else:
                result.errors.append({
                    "file": img_data["filename"],
                    "expected": expected_roll,
                    "detected": detected_roll
                })

        print(f"\nRoll Number Detection: {result.correct}/{result.total} = {result.accuracy:.1%}")
        if result.errors:
            print("Errors:")
            for err in result.errors[:5]:
                print(f"  {err['file']}: expected='{err['expected']}' got='{err['detected']}'")

        assert result.accuracy >= threshold, \
            f"Roll number accuracy {result.accuracy:.1%} below threshold {threshold:.0%}"

    def test_answer_detection_accuracy(self, processor, ground_truth, verified_images):
        """Test answer detection accuracy against threshold."""
        threshold = ground_truth["accuracy_thresholds"]["answers"]
        result = AccuracyResult()

        for img_data in verified_images:
            chayan_expected = img_data.get("chayan_answers", [])
            khoj_expected = img_data.get("khoj_answers", [])

            if not chayan_expected and not khoj_expected:
                continue

            image_path = IMAGES_DIR / img_data["filename"]
            if not image_path.exists():
                continue

            omr_result = processor.process(str(image_path))
            if not omr_result:
                continue

            # Compare Chayan answers
            chayan_detected = getattr(omr_result, 'chayan_answers', []) or []
            for i, (exp, det) in enumerate(zip(chayan_expected, chayan_detected)):
                result.total += 1
                if exp == det:
                    result.correct += 1
                else:
                    result.errors.append({
                        "file": img_data["filename"],
                        "question": f"Chayan Q{i+1}",
                        "expected": exp,
                        "detected": det
                    })

            # Compare Khoj answers
            khoj_detected = getattr(omr_result, 'khoj_answers', []) or []
            for i, (exp, det) in enumerate(zip(khoj_expected, khoj_detected)):
                result.total += 1
                if exp == det:
                    result.correct += 1
                else:
                    result.errors.append({
                        "file": img_data["filename"],
                        "question": f"Khoj Q{i+1}",
                        "expected": exp,
                        "detected": det
                    })

        print(f"\nAnswer Detection: {result.correct}/{result.total} = {result.accuracy:.1%}")
        if result.errors:
            print(f"Errors ({len(result.errors)} total, showing first 5):")
            for err in result.errors[:5]:
                print(f"  {err['file']} {err['question']}: expected='{err['expected']}' got='{err['detected']}'")

        assert result.accuracy >= threshold, \
            f"Answer accuracy {result.accuracy:.1%} below threshold {threshold:.0%}"

    def _names_match(self, detected: str, expected: str) -> bool:
        """Compare names with some tolerance for spacing differences."""
        if not detected or not expected:
            return detected == expected

        # Normalize: remove extra spaces, compare
        detected_norm = ' '.join(detected.split())
        expected_norm = ' '.join(expected.split())

        if detected_norm == expected_norm:
            return True

        # Allow minor character differences (e.g., 'O U' vs 'OU')
        detected_compact = detected_norm.replace(' ', '')
        expected_compact = expected_norm.replace(' ', '')

        return detected_compact == expected_compact


class TestSpecificEdgeCases:
    """Test specific known edge cases that were fixed."""

    @pytest.fixture(scope="class")
    def processor(self):
        from src.ai.yolo_omr_processor import YOLOOMRProcessor
        return YOLOOMRProcessor()

    @pytest.mark.skipif(
        not (project_root / "filledomrsheet" / "41_716.jpg").exists(),
        reason="Test image not found"
    )
    def test_41_716_first_letter_detection(self, processor):
        """Test that 41_716.jpg detects 'A' as first letter (not 'G')."""
        image_path = project_root / "filledomrsheet" / "41_716.jpg"
        result = processor.process(str(image_path))

        assert result is not None, "Failed to process image"
        assert result.name, "No name detected"

        # First letter should be 'A', not 'G' (the bug we fixed)
        first_letter = result.name.strip().split()[0] if result.name else ""
        assert first_letter == 'A', \
            f"First letter should be 'A' but got '{first_letter}'. Full name: '{result.name}'"

    @pytest.mark.skipif(
        not (project_root / "filledomrsheet" / "41_794.jpg").exists(),
        reason="Test image not found"
    )
    def test_41_794_answer_detection(self, processor):
        """Test that 41_794.jpg has consistent answer detection."""
        image_path = project_root / "filledomrsheet" / "41_794.jpg"
        result = processor.process(str(image_path))

        assert result is not None, "Failed to process image"

        # At minimum, should have some answers detected
        khoj_answers = getattr(result, 'khoj_answers', [])
        assert khoj_answers, "No Khoj answers detected"
        assert len(khoj_answers) >= 10, f"Expected 10 Khoj answers, got {len(khoj_answers)}"


def run_quick_validation():
    """Run a quick validation without pytest (for command-line use)."""
    from src.ai.yolo_omr_processor import YOLOOMRProcessor

    print("=" * 70)
    print("Quick Regression Validation")
    print("=" * 70)

    # Load ground truth
    data = load_ground_truth()
    if not data:
        print("ERROR: No ground truth file found")
        print(f"Expected at: {GROUND_TRUTH_FILE}")
        print("\nTo create ground truth:")
        print("  python scripts/create_ground_truth.py")
        return False

    images = get_verified_images(data)
    if not images:
        print("No verified images in ground truth")
        return False

    print(f"Found {len(images)} verified images")

    # Initialize processor
    processor = YOLOOMRProcessor()

    # Run validation
    name_correct = 0
    roll_correct = 0
    total = 0

    for img_data in images:
        image_path = IMAGES_DIR / img_data["filename"]
        if not image_path.exists():
            print(f"  SKIP: {img_data['filename']} (not found)")
            continue

        result = processor.process(str(image_path))
        if not result:
            print(f"  FAIL: {img_data['filename']} (process failed)")
            continue

        total += 1
        name_ok = (result.name or "") == img_data.get("name", "")
        roll_ok = (result.roll_number or "") == img_data.get("roll_number", "")

        if name_ok:
            name_correct += 1
        if roll_ok:
            roll_correct += 1

        status = "OK" if (name_ok and roll_ok) else "MISMATCH"
        if status != "OK":
            print(f"  {status}: {img_data['filename']}")
            if not name_ok:
                print(f"    Name: '{result.name}' vs expected '{img_data.get('name')}'")
            if not roll_ok:
                print(f"    Roll: '{result.roll_number}' vs expected '{img_data.get('roll_number')}'")

    print("\n" + "=" * 70)
    print(f"Results: {total} images tested")
    print(f"  Name accuracy:        {name_correct}/{total} = {name_correct/total:.1%}" if total else "N/A")
    print(f"  Roll number accuracy: {roll_correct}/{total} = {roll_correct/total:.1%}" if total else "N/A")

    thresholds = data.get("accuracy_thresholds", {})
    name_threshold = thresholds.get("name", 0.95)
    roll_threshold = thresholds.get("roll_number", 0.99)

    name_pass = (name_correct / total >= name_threshold) if total else False
    roll_pass = (roll_correct / total >= roll_threshold) if total else False

    print(f"\nThresholds:")
    print(f"  Name:        {name_threshold:.0%} -> {'PASS' if name_pass else 'FAIL'}")
    print(f"  Roll number: {roll_threshold:.0%} -> {'PASS' if roll_pass else 'FAIL'}")

    return name_pass and roll_pass


if __name__ == "__main__":
    # Allow running directly for quick validation
    import sys
    success = run_quick_validation()
    sys.exit(0 if success else 1)

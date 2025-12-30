"""
Regression Tests for OMR Detection

This test suite validates detection accuracy against the golden dataset.
It ensures that bug fixes don't break previously working detections.

Run with: pytest tests/test_regression.py -v
Or directly: python tests/test_regression.py
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, field
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
    errors: List[Dict[str, Any]] = field(default_factory=list)

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


def get_all_images(data: Dict) -> List[Dict]:
    """Get all images from ground truth (verified or not)."""
    return data.get("images", [])


def names_match(detected: str, expected: str) -> bool:
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

    # Use all images (not just verified) for initial testing
    images = get_all_images(data)
    if not images:
        print("No images in ground truth")
        return False

    print(f"Found {len(images)} images in ground truth")

    # Initialize processor
    print("\nInitializing OMR processor...")
    processor = YOLOOMRProcessor()

    # Run validation
    name_correct = 0
    roll_correct = 0
    total = 0
    mismatches = []

    print("\nProcessing images...")
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
        name_ok = names_match(result.name or "", img_data.get("name", ""))
        roll_ok = (result.roll_number or "") == img_data.get("roll_number", "")

        if name_ok:
            name_correct += 1
        if roll_ok:
            roll_correct += 1

        if not name_ok or not roll_ok:
            mismatches.append({
                "file": img_data["filename"],
                "name_expected": img_data.get("name", ""),
                "name_detected": result.name or "",
                "roll_expected": img_data.get("roll_number", ""),
                "roll_detected": result.roll_number or "",
                "name_ok": name_ok,
                "roll_ok": roll_ok
            })

    print("\n" + "=" * 70)
    print(f"Results: {total} images tested")

    if total > 0:
        name_acc = name_correct / total
        roll_acc = roll_correct / total
        print(f"  Name accuracy:        {name_correct}/{total} = {name_acc:.1%}")
        print(f"  Roll number accuracy: {roll_correct}/{total} = {roll_acc:.1%}")

        thresholds = data.get("accuracy_thresholds", {})
        name_threshold = thresholds.get("name", 0.95)
        roll_threshold = thresholds.get("roll_number", 0.99)

        name_pass = name_acc >= name_threshold
        roll_pass = roll_acc >= roll_threshold

        print(f"\nThresholds:")
        print(f"  Name:        {name_threshold:.0%} -> {'PASS' if name_pass else 'FAIL'}")
        print(f"  Roll number: {roll_threshold:.0%} -> {'PASS' if roll_pass else 'FAIL'}")

        if mismatches:
            print(f"\nMismatches ({len(mismatches)} total, showing first 10):")
            for m in mismatches[:10]:
                issues = []
                if not m["name_ok"]:
                    issues.append(f"name: '{m['name_detected']}' vs '{m['name_expected']}'")
                if not m["roll_ok"]:
                    issues.append(f"roll: '{m['roll_detected']}' vs '{m['roll_expected']}'")
                print(f"  {m['file']}: {', '.join(issues)}")

        return name_pass and roll_pass
    else:
        print("No images were tested!")
        return False


# Pytest-based tests (only loaded if pytest is available)
try:
    import pytest

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
                pytest.skip("No ground truth file found.")
            return data

        @pytest.fixture(scope="class")
        def verified_images(self, ground_truth):
            """Get verified images from ground truth."""
            images = get_verified_images(ground_truth)
            if not images:
                pytest.skip("No verified images in ground truth.")
            return images

        def test_ground_truth_exists(self, ground_truth):
            """Verify that ground truth file exists and is valid."""
            assert ground_truth is not None
            assert "images" in ground_truth
            assert "accuracy_thresholds" in ground_truth

        def test_name_detection_accuracy(self, processor, ground_truth, verified_images):
            """Test name detection accuracy against threshold."""
            threshold = ground_truth["accuracy_thresholds"]["name"]
            result = AccuracyResult()

            for img_data in verified_images:
                expected_name = img_data.get("name", "")
                if not expected_name:
                    continue

                image_path = IMAGES_DIR / img_data["filename"]
                if not image_path.exists():
                    continue

                omr_result = processor.process(str(image_path))
                detected_name = omr_result.name if omr_result else ""

                result.total += 1
                if names_match(detected_name, expected_name):
                    result.correct += 1

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

            assert result.accuracy >= threshold, \
                f"Roll accuracy {result.accuracy:.1%} below threshold {threshold:.0%}"

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

            first_letter = result.name.strip().split()[0] if result.name else ""
            assert first_letter == 'A', \
                f"First letter should be 'A' but got '{first_letter}'"

        @pytest.mark.skipif(
            not (project_root / "filledomrsheet" / "41_794.jpg").exists(),
            reason="Test image not found"
        )
        def test_41_794_answer_detection(self, processor):
            """Test that 41_794.jpg has consistent answer detection."""
            image_path = project_root / "filledomrsheet" / "41_794.jpg"
            result = processor.process(str(image_path))

            assert result is not None, "Failed to process image"
            khoj_answers = getattr(result, 'khoj_answers', [])
            assert khoj_answers, "No Khoj answers detected"
            assert len(khoj_answers) >= 10

except ImportError:
    # pytest not available, skip class definitions
    pass


if __name__ == "__main__":
    success = run_quick_validation()
    sys.exit(0 if success else 1)

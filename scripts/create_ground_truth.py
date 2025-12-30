"""
Ground Truth Annotation Helper

This script helps create and update the golden dataset ground truth file.
It runs OMR detection on images and allows you to verify/correct the results.

Usage:
    python scripts/create_ground_truth.py                    # Add all images from golden_dataset/images/
    python scripts/create_ground_truth.py path/to/image.jpg  # Add specific image
    python scripts/create_ground_truth.py --verify           # Verify existing annotations
"""

import sys
import os
import json
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ai.yolo_omr_processor import YOLOOMRProcessor

GOLDEN_DIR = project_root / "golden_dataset"
IMAGES_DIR = GOLDEN_DIR / "images"
GROUND_TRUTH_FILE = GOLDEN_DIR / "ground_truth.json"


def load_ground_truth():
    """Load existing ground truth file."""
    if GROUND_TRUTH_FILE.exists():
        with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "version": "1.0",
        "description": "Golden dataset for OMR detection regression testing",
        "accuracy_thresholds": {
            "name": 0.95,
            "roll_number": 0.99,
            "answers": 0.98
        },
        "images": []
    }


def save_ground_truth(data):
    """Save ground truth file."""
    with open(GROUND_TRUTH_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {GROUND_TRUTH_FILE}")


def get_existing_filenames(data):
    """Get set of already annotated filenames."""
    return {img["filename"] for img in data["images"]}


def annotate_image(processor, image_path: Path, copy_to_golden: bool = True):
    """Run detection and allow user to verify/correct results."""
    print("\n" + "=" * 70)
    print(f"Processing: {image_path.name}")
    print("=" * 70)

    # Run detection
    result = processor.process(str(image_path))

    if not result:
        print("ERROR: Failed to process image")
        return None

    # Display detected values
    print("\nDetected values:")
    print(f"  Name:        {result.name}")
    print(f"  Roll Number: {result.roll_number}")
    print(f"  Class:       {result.student_class}")
    print(f"  Section:     {result.section}")
    print(f"  School Code: {result.school_code}")

    # Format answers
    chayan_answers = result.chayan_answers if hasattr(result, 'chayan_answers') else []
    khoj_answers = result.khoj_answers if hasattr(result, 'khoj_answers') else []

    if not chayan_answers and hasattr(result, 'answers'):
        # Split answers into Chayan (1-10) and Khoj (11-20)
        all_answers = result.answers or []
        chayan_answers = all_answers[:10] if len(all_answers) >= 10 else all_answers
        khoj_answers = all_answers[10:20] if len(all_answers) >= 20 else []

    print(f"  Chayan (Q1-10): {' '.join(chayan_answers) if chayan_answers else 'N/A'}")
    print(f"  Khoj (Q11-20):  {' '.join(khoj_answers) if khoj_answers else 'N/A'}")

    print("\n" + "-" * 70)
    print("Options:")
    print("  [Enter] Accept as correct")
    print("  [n]     Edit name")
    print("  [r]     Edit roll number")
    print("  [c]     Edit Chayan answers")
    print("  [k]     Edit Khoj answers")
    print("  [s]     Skip this image")
    print("  [q]     Quit")

    # Create annotation record
    annotation = {
        "filename": image_path.name,
        "name": result.name or "",
        "roll_number": result.roll_number or "",
        "class": result.student_class or "",
        "section": result.section or "",
        "school_code": result.school_code or "",
        "chayan_answers": chayan_answers,
        "khoj_answers": khoj_answers,
        "scan_quality": "unknown",
        "edge_cases": [],
        "verified": False
    }

    while True:
        choice = input("\nYour choice: ").strip().lower()

        if choice == '' or choice == 'y':
            # Accept current values
            annotation["verified"] = True
            break
        elif choice == 'n':
            new_name = input(f"  Enter correct name [{annotation['name']}]: ").strip()
            if new_name:
                annotation["name"] = new_name
        elif choice == 'r':
            new_roll = input(f"  Enter correct roll number [{annotation['roll_number']}]: ").strip()
            if new_roll:
                annotation["roll_number"] = new_roll
        elif choice == 'c':
            print(f"  Current Chayan answers: {' '.join(annotation['chayan_answers'])}")
            new_answers = input("  Enter 10 answers (space-separated, e.g., A B C D A B C D A B): ").strip().upper()
            if new_answers:
                annotation["chayan_answers"] = new_answers.split()[:10]
        elif choice == 'k':
            print(f"  Current Khoj answers: {' '.join(annotation['khoj_answers'])}")
            new_answers = input("  Enter 10 answers (space-separated, e.g., A C B A B C A C A D): ").strip().upper()
            if new_answers:
                annotation["khoj_answers"] = new_answers.split()[:10]
        elif choice == 's':
            print("  Skipping...")
            return None
        elif choice == 'q':
            print("  Quitting...")
            sys.exit(0)
        else:
            print("  Invalid choice. Try again.")
            continue

        # Show updated values after edit
        if choice in ['n', 'r', 'c', 'k']:
            print("\n  Updated values:")
            print(f"    Name:        {annotation['name']}")
            print(f"    Roll Number: {annotation['roll_number']}")
            print(f"    Chayan:      {' '.join(annotation['chayan_answers'])}")
            print(f"    Khoj:        {' '.join(annotation['khoj_answers'])}")
            print("\n  [Enter] to accept, or continue editing...")

    # Ask about scan quality
    print("\nScan quality:")
    print("  [1] Clean (good contrast, straight)")
    print("  [2] Dark (over-inked)")
    print("  [3] Light (faded)")
    print("  [4] Skewed (rotated)")
    print("  [5] Smudged")
    print("  [6] Other")

    quality_choice = input("Quality [1-6, default=1]: ").strip()
    quality_map = {
        "1": "clean", "2": "dark", "3": "light",
        "4": "skewed", "5": "smudged", "6": "other"
    }
    annotation["scan_quality"] = quality_map.get(quality_choice, "clean")

    # Ask about edge cases
    print("\nEdge cases (comma-separated, or Enter to skip):")
    print("  Examples: multi_fill, light_fill, erasure, empty_name, smudge")
    edge_cases = input("Edge cases: ").strip()
    if edge_cases:
        annotation["edge_cases"] = [e.strip() for e in edge_cases.split(",")]

    # Copy image to golden dataset if needed
    if copy_to_golden:
        dest = IMAGES_DIR / image_path.name
        if not dest.exists():
            shutil.copy2(image_path, dest)
            print(f"\nCopied image to {dest}")

    annotation["verified"] = True
    return annotation


def main():
    args = sys.argv[1:]

    # Create directories if needed
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize processor
    print("Initializing OMR processor...")
    processor = YOLOOMRProcessor()

    # Load existing ground truth
    data = load_ground_truth()
    existing = get_existing_filenames(data)

    if "--verify" in args:
        # Verify mode: re-run detection on existing annotations
        print("\n=== VERIFICATION MODE ===")
        print(f"Found {len(data['images'])} existing annotations")

        for annotation in data["images"]:
            image_path = IMAGES_DIR / annotation["filename"]
            if not image_path.exists():
                print(f"\nWARNING: Image not found: {annotation['filename']}")
                continue

            result = processor.process(str(image_path))
            if result:
                # Compare detected vs ground truth
                name_match = (result.name or "") == annotation.get("name", "")
                roll_match = (result.roll_number or "") == annotation.get("roll_number", "")

                status = "OK" if (name_match and roll_match) else "MISMATCH"
                print(f"\n{annotation['filename']}: {status}")

                if not name_match:
                    print(f"  Name: detected='{result.name}' expected='{annotation.get('name')}'")
                if not roll_match:
                    print(f"  Roll: detected='{result.roll_number}' expected='{annotation.get('roll_number')}'")

        return

    # Determine images to process
    if args and not args[0].startswith("--"):
        # Process specific image
        image_path = Path(args[0])
        if not image_path.exists():
            print(f"ERROR: Image not found: {image_path}")
            sys.exit(1)

        images_to_process = [image_path]
        copy_to_golden = True
    else:
        # Process all images in golden_dataset/images/
        images_to_process = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))

        # Filter out already annotated
        images_to_process = [p for p in images_to_process if p.name not in existing]
        copy_to_golden = False

        if not images_to_process:
            print("\nNo new images to annotate.")
            print(f"Existing annotations: {len(existing)}")
            print(f"\nTo add new images:")
            print(f"  1. Copy images to {IMAGES_DIR}")
            print(f"  2. Run this script again")
            print(f"\nOr specify an image path:")
            print(f"  python scripts/create_ground_truth.py path/to/image.jpg")
            return

    print(f"\nFound {len(images_to_process)} images to annotate")
    print(f"Existing annotations: {len(existing)}")

    # Process each image
    new_count = 0
    for image_path in images_to_process:
        annotation = annotate_image(processor, image_path, copy_to_golden)

        if annotation:
            # Check if updating existing
            existing_idx = next((i for i, img in enumerate(data["images"])
                               if img["filename"] == annotation["filename"]), None)

            if existing_idx is not None:
                data["images"][existing_idx] = annotation
                print(f"  Updated existing annotation")
            else:
                data["images"].append(annotation)
                new_count += 1
                print(f"  Added new annotation")

            # Save after each annotation
            save_ground_truth(data)

    print(f"\n{'=' * 70}")
    print(f"Done! Added {new_count} new annotations.")
    print(f"Total annotations: {len(data['images'])}")
    print(f"Ground truth file: {GROUND_TRUTH_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Training script for improving name detection using ViT model.
Compares current detector output with ground truth from manual verification.
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ai.yolo_omr_processor import YOLOOMRProcessor

# Ground truth data collected from manual image inspection
GROUND_TRUTH = {
    "41_629.jpg": "REYANSH KUMAR",
    "41_632.jpg": "YUVRAJ KUMAR MANDAL",
    "41_635.jpg": "ANJALI KUMARI",
    "41_637.jpg": "ROSHAN RAJWAR",
    "41_642.jpg": "SIMANDHYA KUMARI",
    "41_643.jpg": "RAJ KUMAR",
    "41_645.jpg": "ADJNSH KUMAR NAHTO",
    "41_646.jpg": "RAM KUMAR RAWANI",
    "41_648.jpg": "AYUSH KUMAR YADAV",
    "41_649.jpg": "SRISHTI PRIYA",
    "41_650.jpg": "SAKSHAM KUMARI",
    "41_667.jpg": "SREPTI SHARMA",
    "41_672.jpg": "MOHIT KUMAR DAS",
    "41_676.jpg": "RONAK KUMAR MANDAL",
    "41_679.jpg": "PRINCE KUMAR MANDAL",
    "41_682.jpg": "PRINCE KUMAR MANDAL",
    "41_686.jpg": "AJIT KUMAR YADAV",
    "41_687.jpg": "SUMIT KUMAR TURI",
    "41_702.jpg": "SAGAR KUMAR MANDAL",
    "41_705.jpg": "ISHWAR KUMAR",
    "41_724.jpg": "ADARSH KUMAR",
    "41_726.jpg": "SRISHTI SEN",
    "41_739.jpg": "SONAKSHI KUMARI",
    "41_742.jpg": "KHUSHBU KUMARI",
    "41_744.jpg": "PIHU KUMARI",
    "41_749.jpg": "ASHISH KUMAR MANDAL",
    "41_752.jpg": "KHUSHOV MARANDI",
    "41_753.jpg": "JAYASH PRAKHAR",
    "41_756.jpg": "UMA SIMHA",
    "41_760.jpg": "RAVINDRA PAL",
    "41_768.jpg": "KAVYA KUMARI",
    "41_770.jpg": "ANUPA KUSHWAHA",
    "41_771.jpg": "SHRISHTI KUMARI",
    "41_783.jpg": "PRINCE KUMAR",
    "41_784.jpg": "ANKIT KUMAR",
    "41_785.jpg": "SHASHI KUMARI",
    "41_793.jpg": "LADALI KHAN",
    "41_795.jpg": "SHRISHTI KUMARI",
    "41_798.jpg": "SARA ARFIN",
    "41_810.jpg": "KAUSHAL YADAV",
    "41_815.jpg": "AKSHAY KUMAR",
    "41_820.jpg": "PRATIK SONI",
    "41_826.jpg": "RENU KUMARI",
    "41_830.jpg": "JIVIKA KUMARI",
    "41_841.jpg": "TRISHA SONI",
    "41_850.jpg": "FAIZAN RAJA",
    "41_860.jpg": "SNEHA KUMARI",
    "41_870.jpg": "SNEHA KUMARI",
    "41_880.jpg": "IMRAN KHAN",
    "41_890.jpg": "VISHAL KUMAR",
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def similarity_score(detected: str, ground_truth: str) -> float:
    """Calculate similarity score between detected and ground truth names."""
    if not detected or not ground_truth:
        return 0.0

    # Normalize both strings
    detected = detected.upper().strip()
    ground_truth = ground_truth.upper().strip()

    if detected == ground_truth:
        return 100.0

    # Calculate Levenshtein-based similarity
    max_len = max(len(detected), len(ground_truth))
    if max_len == 0:
        return 0.0

    distance = levenshtein_distance(detected, ground_truth)
    return max(0, (1 - distance / max_len)) * 100


def analyze_errors(detected: str, ground_truth: str) -> Dict:
    """Analyze the type of error between detected and ground truth."""
    detected = detected.upper().strip() if detected else ""
    ground_truth = ground_truth.upper().strip()

    errors = {
        "missing_chars": [],
        "extra_chars": [],
        "wrong_chars": [],
        "truncated": False,
        "partial_match": False
    }

    # Check if detected is a prefix/suffix of ground truth
    if ground_truth.startswith(detected) or ground_truth.endswith(detected):
        errors["truncated"] = True

    # Check if detected contains parts of ground truth
    gt_parts = ground_truth.split()
    det_parts = detected.split() if detected else []

    matching_parts = sum(1 for p in det_parts if p in gt_parts)
    if matching_parts > 0 and matching_parts < len(gt_parts):
        errors["partial_match"] = True

    # Character-level analysis
    gt_chars = set(ground_truth.replace(" ", ""))
    det_chars = set(detected.replace(" ", ""))

    errors["missing_chars"] = list(gt_chars - det_chars)
    errors["extra_chars"] = list(det_chars - gt_chars)

    return errors


def run_comparison(image_dir: str) -> List[Dict]:
    """Run the current detector on all ground truth images and compare."""
    processor = YOLOOMRProcessor()
    results = []

    print("=" * 80)
    print("NAME DETECTION COMPARISON: Current Detector vs Ground Truth")
    print("=" * 80)

    total = len(GROUND_TRUTH)
    correct = 0
    partial = 0

    for filename, ground_truth in GROUND_TRUTH.items():
        image_path = os.path.join(image_dir, filename)

        if not os.path.exists(image_path):
            print(f"[SKIP] {filename} - File not found")
            continue

        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            print(f"[SKIP] {filename} - Could not load image")
            continue

        # Run detector
        omr_result = processor.process(image)
        detected_name = omr_result.name if omr_result.success else ""

        # Calculate similarity
        sim_score = similarity_score(detected_name, ground_truth)
        errors = analyze_errors(detected_name, ground_truth)

        # Determine status
        if sim_score >= 95:
            status = "CORRECT"
            correct += 1
        elif sim_score >= 70:
            status = "PARTIAL"
            partial += 1
        else:
            status = "WRONG"

        result = {
            "filename": filename,
            "ground_truth": ground_truth,
            "detected": detected_name,
            "similarity": sim_score,
            "status": status,
            "errors": errors
        }
        results.append(result)

        # Print result
        status_icon = {"CORRECT": "[OK]", "PARTIAL": "[~]", "WRONG": "[X]"}[status]
        print(f"{status_icon} {filename}")
        print(f"    Ground Truth: {ground_truth}")
        print(f"    Detected:     {detected_name}")
        print(f"    Similarity:   {sim_score:.1f}%")
        if status != "CORRECT":
            if errors["truncated"]:
                print(f"    Issue: Name truncated")
            if errors["partial_match"]:
                print(f"    Issue: Partial match (missing parts)")
            if errors["missing_chars"]:
                print(f"    Missing chars: {errors['missing_chars']}")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total images:     {total}")
    print(f"Correct (>=95%):  {correct} ({100*correct/total:.1f}%)")
    print(f"Partial (70-95%): {partial} ({100*partial/total:.1f}%)")
    print(f"Wrong (<70%):     {total - correct - partial} ({100*(total-correct-partial)/total:.1f}%)")

    avg_similarity = sum(r["similarity"] for r in results) / len(results) if results else 0
    print(f"Average similarity: {avg_similarity:.1f}%")

    return results


def extract_bubble_training_data(image_dir: str, output_dir: str):
    """Extract bubble images with labels for training."""
    processor = YOLOOMRProcessor()

    os.makedirs(output_dir, exist_ok=True)

    training_data = []

    for filename, ground_truth in GROUND_TRUTH.items():
        image_path = os.path.join(image_dir, filename)

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract name field bubbles
        config = processor.FIELD_CONFIGS['name']
        origin = config['origin']
        bubble_size = config['bubble_size']
        rows_gap = config['rows_gap']
        cols_gap = config['cols_gap']
        num_rows = config['rows']
        num_cols = config['cols']
        values = config['values']

        # Parse ground truth into expected bubble positions
        gt_name = ground_truth.upper().replace(" ", "")

        for col in range(min(len(gt_name), num_cols)):
            expected_char = gt_name[col]
            expected_row = values.index(expected_char) if expected_char in values else -1

            x = origin[0] + col * cols_gap

            for row in range(num_rows):
                y = origin[1] + row * rows_gap
                bw, bh = bubble_size

                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble_img = gray[y:y+bh, x:x+bw]

                    # Label: 1 if this is the filled bubble, 0 otherwise
                    is_filled = 1 if row == expected_row else 0

                    # Save bubble image
                    bubble_filename = f"{filename[:-4]}_col{col}_row{row}_{is_filled}.png"
                    bubble_path = os.path.join(output_dir, bubble_filename)
                    cv2.imwrite(bubble_path, bubble_img)

                    # Calculate darkness
                    darkness = np.mean(bubble_img)

                    training_data.append({
                        "filename": bubble_filename,
                        "source_image": filename,
                        "column": col,
                        "row": row,
                        "expected_char": values[row],
                        "is_filled": is_filled,
                        "darkness": float(darkness)
                    })

    # Save training metadata
    with open(os.path.join(output_dir, "training_data.json"), "w") as f:
        json.dump(training_data, f, indent=2)

    # Analyze filled vs unfilled bubble statistics
    filled = [d for d in training_data if d["is_filled"] == 1]
    unfilled = [d for d in training_data if d["is_filled"] == 0]

    print("\n" + "=" * 80)
    print("BUBBLE TRAINING DATA STATISTICS")
    print("=" * 80)
    print(f"Total bubbles extracted: {len(training_data)}")
    print(f"Filled bubbles: {len(filled)}")
    print(f"Unfilled bubbles: {len(unfilled)}")

    if filled:
        avg_filled = sum(d["darkness"] for d in filled) / len(filled)
        min_filled = min(d["darkness"] for d in filled)
        max_filled = max(d["darkness"] for d in filled)
        print(f"\nFilled bubble darkness: avg={avg_filled:.1f}, min={min_filled:.1f}, max={max_filled:.1f}")

    if unfilled:
        avg_unfilled = sum(d["darkness"] for d in unfilled) / len(unfilled)
        min_unfilled = min(d["darkness"] for d in unfilled)
        max_unfilled = max(d["darkness"] for d in unfilled)
        print(f"Unfilled bubble darkness: avg={avg_unfilled:.1f}, min={min_unfilled:.1f}, max={max_unfilled:.1f}")

    # Find problematic overlaps
    if filled and unfilled:
        overlap_threshold = max_filled
        problematic = [d for d in unfilled if d["darkness"] <= overlap_threshold]
        print(f"\nProblematic unfilled bubbles (darkness <= {overlap_threshold:.1f}): {len(problematic)}")

    print(f"\nTraining data saved to: {output_dir}")

    return training_data


def main():
    """Main function."""
    image_dir = "/Volumes/Projects/Projects/omrsheetproj/filledomrsheet"
    output_dir = "/Volumes/Projects/Projects/omrsheetproj/bubble_training_data"

    print("\n" + "=" * 80)
    print("STEP 1: Running comparison between detector and ground truth")
    print("=" * 80 + "\n")

    results = run_comparison(image_dir)

    # Save comparison results
    with open("name_detection_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nComparison results saved to: name_detection_comparison.json")

    print("\n" + "=" * 80)
    print("STEP 2: Extracting bubble training data")
    print("=" * 80 + "\n")

    training_data = extract_bubble_training_data(image_dir, output_dir)

    # Identify common error patterns
    print("\n" + "=" * 80)
    print("STEP 3: Error Pattern Analysis")
    print("=" * 80 + "\n")

    wrong_results = [r for r in results if r["status"] == "WRONG"]
    partial_results = [r for r in results if r["status"] == "PARTIAL"]

    if wrong_results or partial_results:
        print("Common issues found:")

        truncated = sum(1 for r in results if r["errors"]["truncated"])
        partial_match = sum(1 for r in results if r["errors"]["partial_match"])

        print(f"  - Truncated names: {truncated}")
        print(f"  - Partial matches: {partial_match}")

        # Collect all missing characters
        all_missing = []
        for r in results:
            all_missing.extend(r["errors"]["missing_chars"])

        if all_missing:
            from collections import Counter
            char_counts = Counter(all_missing)
            print(f"\n  Most commonly missed characters:")
            for char, count in char_counts.most_common(10):
                print(f"    '{char}': {count} times")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

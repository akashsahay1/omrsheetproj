#!/usr/bin/env python3
"""
Test bubble-based name detection with calibrated coordinates.
"""

import os
import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ai.yolo_omr_processor import YOLOOMRProcessor

# Ground truth data from manual verification
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
    if not detected or not ground_truth:
        return 0.0

    detected = detected.upper().strip()
    ground_truth = ground_truth.upper().strip()

    if detected == ground_truth:
        return 100.0

    max_len = max(len(detected), len(ground_truth))
    if max_len == 0:
        return 0.0

    distance = levenshtein_distance(detected, ground_truth)
    return max(0, (1 - distance / max_len)) * 100


def test_bubble_detection():
    """Test bubble-based name detection."""
    # Use cross-platform path relative to this script
    project_dir = Path(__file__).parent
    image_dir = project_dir / "filledomrsheet"

    processor = YOLOOMRProcessor()

    print("=" * 80)
    print("BUBBLE-BASED NAME DETECTION TEST")
    print("=" * 80)

    total = 0
    correct = 0
    partial = 0
    results = []

    for filename, ground_truth in GROUND_TRUTH.items():
        image_path = image_dir / filename

        if not image_path.exists():
            print(f"[SKIP] {filename} - not found")
            continue

        total += 1

        # Process with bubble detection (pass file path, not image)
        result = processor.process(str(image_path))
        if not result.success:
            print(f"[SKIP] {filename} - processing failed: {result.error_message}")
            continue

        detected = result.name

        # Calculate similarity
        sim = similarity_score(detected, ground_truth)

        # Determine status
        if sim >= 95:
            status = "OK"
            correct += 1
        elif sim >= 70:
            status = "PARTIAL"
            partial += 1
        else:
            status = "WRONG"

        results.append({
            "filename": filename,
            "ground_truth": ground_truth,
            "detected": detected,
            "similarity": sim,
            "status": status
        })

        icon = {"OK": "✓", "PARTIAL": "~", "WRONG": "✗"}[status]
        print(f"[{icon}] {filename}")
        print(f"    GT:  {ground_truth}")
        print(f"    Det: {detected}")
        print(f"    Sim: {sim:.1f}%")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total:   {total}")
    print(f"Correct: {correct} ({100*correct/total:.1f}%)")
    print(f"Partial: {partial} ({100*partial/total:.1f}%)")
    print(f"Wrong:   {total - correct - partial} ({100*(total-correct-partial)/total:.1f}%)")

    avg_sim = sum(r["similarity"] for r in results) / len(results) if results else 0
    print(f"Average similarity: {avg_sim:.1f}%")

    return results


if __name__ == "__main__":
    test_bubble_detection()

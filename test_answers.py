#!/usr/bin/env python3
"""Test answer detection accuracy."""
import cv2
from pathlib import Path
from src.ai.yolo_omr_processor import YOLOOMRProcessor

def test_answers():
    processor = YOLOOMRProcessor()

    # Test on a few images
    images = list(Path("filledomrsheet").glob("*.jpg"))[:10]

    print(f"Testing answer detection on {len(images)} images...\n")
    print(f"{'Filename':<20} {'Answers':<15} {'Count':<8}")
    print("-" * 45)

    total_detected = 0
    for img_path in images:
        result = processor.process(str(img_path))
        if result.success:
            answered = sum(1 for a in result.answers.values() if a)
            total_detected += answered
            # Show first few answers
            sample = [f"Q{k}:{v}" for k, v in list(result.answers.items())[:5] if v]
            print(f"{img_path.name:<20} {','.join(sample)[:15]:<15} {answered}/60")
        else:
            print(f"{img_path.name:<20} FAILED: {result.error_message[:30]}")

    avg_detected = total_detected / len(images) if images else 0
    print(f"\nAverage detected answers: {avg_detected:.1f}/60")

if __name__ == "__main__":
    test_answers()

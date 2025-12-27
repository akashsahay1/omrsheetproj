#!/usr/bin/env python3
"""Trace through name cleaning to find the issue."""

import sys
sys.path.insert(0, '/Volumes/Projects/Projects/omrsheetproj')

from src.ai.yolo_omr_processor import YOLOOMRProcessor

def test_cleaning():
    processor = YOLOOMRProcessor()

    # Test cases from detection
    test_cases = [
        "KARAN MAHTO",      # From detection with space
        "KARANMAHTO",       # Without space
        "ANJALIKUMRI",      # Missing space
        "VIVEKMOHTO",       # Wrong letters + no space
    ]

    for raw in test_cases:
        cleaned = processor._clean_detected_name(raw)
        print(f"{raw:25s} -> {cleaned}")

if __name__ == "__main__":
    test_cleaning()

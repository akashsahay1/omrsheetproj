"""
AI-based OMR Detection Module

This module provides AI-powered OMR sheet scanning that doesn't rely on
fixed coordinates, making it robust to:
- Scanner DPI variations
- Page positioning differences
- Skew and rotation
- Scale changes

Architecture:
1. MarkerAligner - Detects corner markers and warps to normalized space
2. BubbleDetector - YOLO-based bubble detection (filled vs empty)
3. TextRecognizer - TrOCR-based handwritten text recognition
"""

from .marker_aligner import MarkerAligner
from .bubble_detector import AIBubbleDetector
from .text_recognizer import TextRecognizer
from .omr_ai_engine import OMRAIEngine

__all__ = [
    'MarkerAligner',
    'AIBubbleDetector',
    'TextRecognizer',
    'OMRAIEngine'
]

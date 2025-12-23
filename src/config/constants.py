"""
Application constants and enumerations
"""
from enum import Enum
from typing import List

# Application Info
APP_NAME = "OMR Processor"
APP_VERSION = "1.0.0"
APP_TITLE = "OMR Processor - Vidya Bharti Pratibha Khoj 2025-26"

# Supported Image Formats
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']

# OMR Sheet Specifications
class SheetConfig:
    """Configuration for Vidya Bharti OMR Sheet"""

    # Image dimensions (expected after scanning)
    PAGE_WIDTH = 2480
    PAGE_HEIGHT = 3507

    # Bubble dimensions
    BUBBLE_WIDTH = 32
    BUBBLE_HEIGHT = 32

    # Name grid configuration
    NAME_COLUMNS = 25  # Maximum 25 characters
    NAME_ROWS = 26     # A-Z letters

    # Roll number configuration
    ROLL_NUMBER_DIGITS = 3

    # School code configuration
    SCHOOL_CODE_DIGITS = 4

    # Answer section configuration
    TOTAL_QUESTIONS = 60
    OPTIONS_PER_QUESTION = 4  # A, B, C, D
    QUESTIONS_PER_COLUMN = 10
    ANSWER_COLUMNS = 6

    # Class options
    CLASS_OPTIONS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Section options
    SECTION_OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    # Answer options
    ANSWER_OPTIONS = ['A', 'B', 'C', 'D']

    # Alphabet for name
    ALPHABET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


class ScoringConfig:
    """Default scoring configuration"""
    CORRECT_MARKS = 1.0
    WRONG_MARKS = -0.25
    UNANSWERED_MARKS = 0.0
    MULTI_MARKED_MARKS = -0.25  # Treat as wrong


class ProcessingStatus(Enum):
    """Status of sheet processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    ALIGNMENT_ERROR = "alignment_error"
    SKIPPED = "skipped"


class BubbleStatus(Enum):
    """Status of individual bubble detection"""
    EMPTY = "empty"
    FILLED = "filled"
    PARTIAL = "partial"  # Partially filled (ambiguous)
    MULTI = "multi"      # Multiple marks detected


# Detection thresholds
class DetectionThresholds:
    """Thresholds for bubble detection"""

    # Mean intensity threshold (0-255, lower = darker = filled)
    # Lower value = more strict, Higher value = more lenient
    FILL_THRESHOLD = 150  # More strict: only count truly dark pixels

    # Minimum fill ratio for a bubble to be considered marked
    # Values from sample: filled=0.42-0.96, empty=0.18-0.32 (outline pixels)
    MIN_FILL_RATIO = 0.40  # Increased to avoid false positives from outlines

    # Partial fill threshold (for flagging ambiguous marks)
    PARTIAL_THRESHOLD = 0.30

    # Alignment marker detection threshold
    MARKER_THRESHOLD = 0.7


# Special values
EMPTY_VALUE = ""
MULTI_MARK_VALUE = "MULTI"
ERROR_VALUE = "ERROR"


# File paths
class DefaultPaths:
    """Default file paths"""
    TEMPLATE_DIR = "templates"
    DEFAULT_TEMPLATE = "vidya_bharti_2025"
    EXPORTS_DIR = "data/exports"
    ANSWER_KEYS_DIR = "data/answer_keys"

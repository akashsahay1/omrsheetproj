"""
AI-powered OMR processing engine.

Combines marker alignment, bubble detection, and text recognition
for robust OMR scanning that doesn't rely on fixed coordinates.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import json

from .marker_aligner import MarkerAligner, AlignmentResult
from .bubble_detector import AIBubbleDetector, BubbleGrid
from .text_recognizer import TextRecognizer


@dataclass
class OMRAIResult:
    """Result of AI-based OMR processing."""
    success: bool
    name: str = ""
    roll_number: str = ""
    class_: str = ""
    section: str = ""
    school_code: str = ""
    answers: Dict[int, str] = None
    confidence: float = 0.0
    error_message: str = ""
    processing_details: Dict[str, Any] = None

    def __post_init__(self):
        if self.answers is None:
            self.answers = {}
        if self.processing_details is None:
            self.processing_details = {}


class OMRAIEngine:
    """
    AI-powered OMR processing engine.

    Pipeline:
    1. Align sheet using marker detection or feature matching
    2. Detect bubbles using AI (YOLO) or traditional CV
    3. Recognize handwritten text using OCR
    4. Extract and validate results
    """

    # Field regions calibrated for Vidya Bharti 2025 OMR sheet
    # Coordinates are for 2480x3507 normalized image
    FIELD_REGIONS = {
        'name': {
            'origin': (193, 690),  # Top-left of first bubble
            'bubble_size': (28, 28),
            'rows_gap': 51,  # Vertical gap between rows (A-Z)
            'cols_gap': 54,  # Horizontal gap between columns
            'type': 'bubble_grid',
            'rows': 26,  # A-Z
            'cols': 25,
            'values': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        },
        'name_handwritten': {
            'region': (100, 300, 1400, 350),  # Handwritten name area above bubbles
            'type': 'text'
        },
        'roll_number': {
            'origin': (1624, 1278),
            'bubble_size': (28, 28),
            'rows_gap': 54,
            'cols_gap': 56,
            'type': 'bubble_grid',
            'rows': 10,  # 0-9
            'cols': 3,
            'values': list('0123456789')
        },
        'class': {
            'origin': (1792, 1278),
            'bubble_size': (28, 28),
            'rows_gap': 54,
            'cols_gap': 56,
            'type': 'bubble_grid',
            'rows': 8,  # 3-10
            'cols': 1,
            'values': ['3', '4', '5', '6', '7', '8', '9', '10']
        },
        'section': {
            'origin': (1848, 1278),
            'bubble_size': (28, 28),
            'rows_gap': 54,
            'cols_gap': 56,
            'type': 'bubble_grid',
            'rows': 8,  # A-H
            'cols': 1,
            'values': list('ABCDEFGH')
        },
        'school_code': {
            'origin': (1904, 1278),
            'bubble_size': (28, 28),
            'rows_gap': 54,
            'cols_gap': 56,
            'type': 'bubble_grid',
            'rows': 10,  # 0-9
            'cols': 3,
            'values': list('0123456789')
        },
        'answers': {
            'type': 'answer_blocks',
            'blocks': [
                {'origin': (167, 1656), 'questions': range(1, 11)},
                {'origin': (557, 1656), 'questions': range(11, 21)},
                {'origin': (947, 1656), 'questions': range(21, 31)},
                {'origin': (1337, 1656), 'questions': range(31, 41)},
                {'origin': (1727, 1656), 'questions': range(41, 51)},
                {'origin': (2117, 1656), 'questions': range(51, 61)},
            ],
            'bubble_size': (28, 28),
            'rows_gap': 54,  # Gap between options A-D
            'cols_gap': 70,  # Gap between questions
            'options': 4,
            'values': list('ABCD')
        }
    }

    def __init__(
        self,
        reference_image_path: Optional[str] = None,
        bubble_model_path: Optional[str] = None,
        ocr_backend: str = "easyocr",
        use_gpu: bool = False
    ):
        """
        Initialize the AI OMR engine.

        Args:
            reference_image_path: Path to reference OMR sheet image for alignment
            bubble_model_path: Path to trained YOLO model for bubble detection
            ocr_backend: OCR backend for text recognition ("trocr", "easyocr", "tesseract")
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu

        # Initialize components
        self.aligner = MarkerAligner()
        self.bubble_detector = AIBubbleDetector(
            model_path=bubble_model_path,
            use_gpu=use_gpu
        )
        self.text_recognizer = TextRecognizer(
            backend=ocr_backend,
            use_gpu=use_gpu
        )

        # Load reference image for alignment
        self.reference_image = None
        if reference_image_path and Path(reference_image_path).exists():
            self.reference_image = cv2.imread(reference_image_path)

    def process(self, image_path: str) -> OMRAIResult:
        """
        Process an OMR sheet image.

        Args:
            image_path: Path to the OMR image

        Returns:
            OMRAIResult with extracted data
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return OMRAIResult(
                    success=False,
                    error_message=f"Failed to load image: {image_path}"
                )

            # Step 1: Align the sheet
            aligned = self._align_sheet(image)
            if aligned is None:
                return OMRAIResult(
                    success=False,
                    error_message="Failed to align sheet"
                )

            # Step 2: Extract fields
            result = OMRAIResult(success=True, processing_details={})

            # Extract name (try both bubble and handwritten)
            bubble_name = self._extract_bubble_field(aligned, 'name')
            handwritten_name = self._extract_text_field(aligned, 'name_handwritten')

            # Use the more confident result
            if len(handwritten_name) > len(bubble_name) * 0.7:
                result.name = handwritten_name
                result.processing_details['name_source'] = 'handwritten'
            else:
                result.name = bubble_name
                result.processing_details['name_source'] = 'bubble'

            # Extract other fields
            result.roll_number = self._extract_bubble_field(aligned, 'roll_number')
            result.class_ = self._extract_bubble_field(aligned, 'class')
            result.section = self._extract_bubble_field(aligned, 'section')
            result.school_code = self._extract_bubble_field(aligned, 'school_code')

            # Extract answers
            result.answers = self._extract_answers(aligned)

            # Calculate overall confidence
            result.confidence = self._calculate_confidence(result)

            return result

        except Exception as e:
            import traceback
            return OMRAIResult(
                success=False,
                error_message=f"Processing error: {str(e)}\n{traceback.format_exc()}"
            )

    def _align_sheet(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Align the OMR sheet to a standard orientation."""
        # Try marker-based alignment first
        result = self.aligner.align(image)

        if result.success:
            return result.aligned_image

        # Fall back to feature-based alignment if we have a reference
        if self.reference_image is not None:
            result = self.aligner.align_with_reference(image, self.reference_image)
            if result.success:
                return result.aligned_image

        # Last resort: just resize to standard dimensions
        return cv2.resize(image, (2480, 3507))

    def _extract_bubble_field(self, image: np.ndarray, field_name: str) -> str:
        """Extract a bubble-based field value using grid sampling."""
        field_config = self.FIELD_REGIONS.get(field_name)
        if not field_config or field_config.get('type') != 'bubble_grid':
            return ""

        # Get grid parameters
        origin = field_config['origin']
        bubble_size = field_config.get('bubble_size', (28, 28))
        rows_gap = field_config.get('rows_gap', 51)
        cols_gap = field_config.get('cols_gap', 54)
        num_rows = field_config.get('rows', 26)
        num_cols = field_config.get('cols', 25)
        values = field_config.get('values', [])

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        result_chars = []

        # For each column, find the darkest (filled) bubble
        for col in range(num_cols):
            x = origin[0] + col * cols_gap

            # Get all bubble values in this column
            bubble_values = []
            for row in range(num_rows):
                y = origin[1] + row * rows_gap

                # Extract bubble region
                bw, bh = bubble_size
                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble = gray[y:y+bh, x:x+bw]
                    darkness = np.mean(bubble)
                    bubble_values.append((row, darkness))

            if not bubble_values:
                result_chars.append(' ')
                continue

            # Find the darkest bubble
            darkest_row, darkest_val = min(bubble_values, key=lambda x: x[1])

            # Calculate median to determine threshold
            all_vals = [v for _, v in bubble_values]
            median_val = np.median(all_vals)

            # Check if significantly darker than median (filled)
            if darkest_val < median_val * 0.75:
                if darkest_row < len(values):
                    result_chars.append(values[darkest_row])
                else:
                    result_chars.append(' ')
            else:
                result_chars.append(' ')

        # Clean up result
        result = ''.join(result_chars).rstrip()
        result = ' '.join(result.split())  # Normalize spaces

        return result

    def _extract_text_field(self, image: np.ndarray, field_name: str) -> str:
        """Extract a text-based field using OCR."""
        field_config = self.FIELD_REGIONS.get(field_name)
        if not field_config or field_config.get('type') != 'text':
            return ""

        region = field_config['region']
        return self.text_recognizer.recognize_name_field(image, region)

    def _extract_answers(self, image: np.ndarray) -> Dict[int, str]:
        """Extract answer bubbles for all questions using grid sampling."""
        answers = {}
        field_config = self.FIELD_REGIONS['answers']

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        bubble_size = field_config.get('bubble_size', (28, 28))
        rows_gap = field_config.get('rows_gap', 54)
        cols_gap = field_config.get('cols_gap', 70)
        num_options = field_config.get('options', 4)
        values = field_config.get('values', list('ABCD'))

        # Process each answer block
        for block in field_config.get('blocks', []):
            origin = block['origin']
            questions = list(block['questions'])

            for q_idx, q_num in enumerate(questions):
                x = origin[0] + q_idx * cols_gap

                # Get darkness values for all options
                option_values = []
                for opt_idx in range(num_options):
                    y = origin[1] + opt_idx * rows_gap

                    bw, bh = bubble_size
                    if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                        bubble = gray[y:y+bh, x:x+bw]
                        darkness = np.mean(bubble)
                        option_values.append((opt_idx, darkness))

                if not option_values:
                    answers[q_num] = ''
                    continue

                # Find the darkest option
                darkest_opt, darkest_val = min(option_values, key=lambda x: x[1])

                # Check if significantly darker than others
                all_vals = [v for _, v in option_values]
                median_val = np.median(all_vals)

                if darkest_val < median_val * 0.75:
                    if darkest_opt < len(values):
                        answers[q_num] = values[darkest_opt]
                    else:
                        answers[q_num] = ''
                else:
                    answers[q_num] = ''

        return answers

    def _calculate_confidence(self, result: OMRAIResult) -> float:
        """Calculate overall confidence score for the result."""
        scores = []

        # Name confidence
        if result.name:
            name_score = min(1.0, len(result.name) / 10)  # Longer names = more confident
            scores.append(name_score)

        # Roll number confidence
        if result.roll_number and result.roll_number.isdigit():
            scores.append(0.9)
        elif result.roll_number:
            scores.append(0.5)

        # Class confidence
        if result.class_ in ['3', '4', '5', '6', '7', '8', '9', '10']:
            scores.append(1.0)
        elif result.class_:
            scores.append(0.3)

        # Section confidence
        if result.section in 'ABCDEFGH':
            scores.append(1.0)
        elif result.section:
            scores.append(0.3)

        return sum(scores) / len(scores) if scores else 0.0

    def process_batch(
        self,
        image_paths: List[str],
        progress_callback=None
    ) -> List[OMRAIResult]:
        """
        Process multiple OMR images.

        Args:
            image_paths: List of image paths
            progress_callback: Optional callback(current, total, result)

        Returns:
            List of OMRAIResult
        """
        results = []
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            result = self.process(path)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, total, result)

        return results


def create_ai_engine(
    config_path: Optional[str] = None,
    use_gpu: bool = False
) -> OMRAIEngine:
    """
    Factory function to create an AI OMR engine with configuration.

    Args:
        config_path: Path to configuration JSON file
        use_gpu: Whether to use GPU acceleration

    Returns:
        Configured OMRAIEngine instance
    """
    reference_image = None
    bubble_model = None
    ocr_backend = "easyocr"

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            config = json.load(f)

        reference_image = config.get('reference_image')
        bubble_model = config.get('bubble_model')
        ocr_backend = config.get('ocr_backend', 'easyocr')

    return OMRAIEngine(
        reference_image_path=reference_image,
        bubble_model_path=bubble_model,
        ocr_backend=ocr_backend,
        use_gpu=use_gpu
    )

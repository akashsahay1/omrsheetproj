"""
Field extraction module for OMR sheets.

Uses GridDetector and AdaptiveBubbleClassifier to extract field values
(name, roll number, class, section, school code, answers) from OMR images.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .grid_detector import GridDetector, GridConfig, GRID_SEARCH_REGIONS
from .bubble_detector import AdaptiveBubbleClassifier, create_adaptive_classifier
from .dynamic_grid_detector import DynamicGridDetector


@dataclass
class ExtractionResult:
    """Result of field extraction."""
    value: str
    confidence: float
    raw_detections: List[Tuple[int, bool, float]] = field(default_factory=list)  # (row, is_filled, conf)


@dataclass
class OMRExtractionResult:
    """Complete extraction result for an OMR sheet."""
    name: str = ""
    roll_number: str = ""
    class_: str = ""
    section: str = ""
    school_code: str = ""
    answers: Dict[int, str] = field(default_factory=dict)

    # Confidence scores
    name_confidence: float = 0.0
    roll_confidence: float = 0.0
    class_confidence: float = 0.0
    section_confidence: float = 0.0
    school_code_confidence: float = 0.0
    answers_confidence: float = 0.0
    overall_confidence: float = 0.0


# Field value mappings
FIELD_VALUES = {
    'name': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),  # 26 letters
    'roll_number': list('0123456789'),  # 10 digits
    'class': ['3', '4', '5', '6', '7', '8', '9', '10'],  # Classes 3-10
    'section': list('ABCDEFGH'),  # Sections A-H
    'school_code': list('0123456789'),  # 10 digits
    'answers': list('ABCD'),  # 4 options
}


class FieldExtractor:
    """
    Extracts field values from OMR sheets using grid detection and ML classification.

    Uses:
    1. GridDetector for finding bubble positions
    2. AdaptiveBubbleClassifier for filled/empty classification
    3. First-filled-wins logic for multi-marked columns
    """

    def __init__(
        self,
        bubble_classifier: Optional[AdaptiveBubbleClassifier] = None,
        grid_detector: Optional[GridDetector] = None
    ):
        """
        Initialize the field extractor.

        Args:
            bubble_classifier: Classifier for bubble detection, or None to auto-create
            grid_detector: Grid detector, or None to auto-create
        """
        self.bubble_classifier = bubble_classifier or create_adaptive_classifier()
        self.grid_detector = grid_detector or GridDetector()

    def extract_all_fields(self, gray: np.ndarray, color_image: np.ndarray = None) -> OMRExtractionResult:
        """
        Extract all fields from an aligned grayscale OMR image.

        Args:
            gray: Grayscale image aligned to canonical frame (2480x3507)
            color_image: Optional BGR color image for batch detection

        Returns:
            OMRExtractionResult with all extracted values
        """
        result = OMRExtractionResult()

        # Detect batch type from color image
        batch_type = 'khoj'  # Default
        if color_image is not None:
            from ai.dynamic_grid_detector import DynamicGridDetector
            detector = DynamicGridDetector()
            batch_type = detector._detect_batch_type(color_image)

        # Extract individual fields (pass batch_type to all)
        name_result = self.extract_name(gray, batch_type)
        result.name = name_result.value
        result.name_confidence = name_result.confidence

        roll_result = self.extract_roll_number(gray, batch_type)
        result.roll_number = roll_result.value
        result.roll_confidence = roll_result.confidence

        class_result = self.extract_class(gray, batch_type)
        result.class_ = class_result.value
        result.class_confidence = class_result.confidence

        section_result = self.extract_section(gray, batch_type)
        result.section = section_result.value
        result.section_confidence = section_result.confidence

        school_result = self.extract_school_code(gray, batch_type)
        result.school_code = school_result.value
        result.school_code_confidence = school_result.confidence

        answers_result = self.extract_answers(gray, color_image, batch_type)
        result.answers = {int(q): ans for q, ans in answers_result.items() if ans}
        result.answers_confidence = self._calculate_answers_confidence(answers_result)

        # Calculate overall confidence (0-100 scale)
        # Note: Individual field confidences from ML classifier are 0-1
        # answers_confidence is already 0-100 after our fix
        confidences = [
            result.name_confidence * 100,  # Scale 0-1 to 0-100
            result.roll_confidence * 100,
            result.class_confidence * 100,
            result.section_confidence * 100,
            result.school_code_confidence * 100,
            result.answers_confidence  # Already 0-100
        ]
        valid_confs = [c for c in confidences if c > 0]
        result.overall_confidence = np.mean(valid_confs) if valid_confs else 0.0

        return result

    def extract_name(self, gray: np.ndarray, batch_type: str = 'khoj') -> ExtractionResult:
        """
        Extract name field (25 characters max).

        Args:
            gray: Grayscale image
            batch_type: 'khoj' or 'chayan' - affects grid spacing

        Returns:
            ExtractionResult with extracted name
        """
        grid = self.grid_detector.detect_grid(gray, 'name')
        if grid is None:
            return ExtractionResult(value="", confidence=0.0)

        # Chayan-specific calibration parameters
        # These were calibrated from visualization testing on 39_18.jpg
        CHAYAN_ORIGIN_Y = 660
        CHAYAN_ROW_GAP = 52  # Increased from 51 to fix Y-drift
        CHAYAN_BUBBLE_SIZE = 28

        # Actual X positions for all 25 columns (with inserted missing columns 7 and 13)
        # Columns 7 and 13 were empty in the test name but exist on the form
        CHAYAN_COL_X = [
            236, 287, 340, 389, 440, 491,  # Columns 1-6 (First name section)
            542,                            # Column 7 (empty column at end of first name)
            590, 641, 692, 741, 794,        # Columns 8-12 (Middle name section)
            845,                            # Column 13 (empty column at end of middle name)
            895, 946, 995, 1048, 1099, 1150,  # Columns 14-19 (Last name section)
            1201, 1252, 1303, 1354, 1405, 1456  # Columns 20-25 (extrapolated)
        ]

        if batch_type == 'chayan':
            grid = GridConfig(
                origin=(236, CHAYAN_ORIGIN_Y),
                rows=26,
                cols=25,
                row_gap=CHAYAN_ROW_GAP,
                col_gap=51,  # Not used directly since we use CHAYAN_COL_X
                bubble_radius=14,
                confidence=0.95
            )

        values = FIELD_VALUES['name']
        result_chars = []
        confidences = []
        all_detections = []

        # Expected 25 columns for name
        for col in range(min(25, grid.cols)):
            if batch_type == 'chayan':
                # Use direct threshold detection with calibrated X positions
                x = CHAYAN_COL_X[col] if col < len(CHAYAN_COL_X) else CHAYAN_COL_X[-1] + (col - len(CHAYAN_COL_X) + 1) * 51
                darkest_val = 255
                darkest_row = -1

                for row in range(26):
                    y = int(CHAYAN_ORIGIN_Y + row * CHAYAN_ROW_GAP)

                    if x < 0 or y < 0 or x + CHAYAN_BUBBLE_SIZE > gray.shape[1] or y + CHAYAN_BUBBLE_SIZE > gray.shape[0]:
                        continue

                    region = gray[y:y+CHAYAN_BUBBLE_SIZE, x:x+CHAYAN_BUBBLE_SIZE]
                    avg = np.mean(region)

                    if avg < darkest_val:
                        darkest_val = avg
                        darkest_row = row

                if darkest_row >= 0 and darkest_val < 180:
                    result_chars.append(values[darkest_row])
                    confidences.append(1.0 - darkest_val / 255)
                else:
                    result_chars.append(' ')
                    confidences.append(0.0)
            else:
                # Use bubble classifier for Khoj (works well for red forms)
                bubble_images = self._extract_column_bubbles(gray, grid, col, 0)

                if not bubble_images:
                    result_chars.append(' ')
                    confidences.append(0.0)
                    continue

                # Classify and select
                char, conf = self.bubble_classifier.classify_and_select(bubble_images, values)
                result_chars.append(char)
                confidences.append(conf)

        # Clean the name
        raw_name = ''.join(result_chars)
        cleaned_name = self._clean_name(raw_name)

        avg_confidence = np.mean(confidences) if confidences else 0.0

        return ExtractionResult(
            value=cleaned_name,
            confidence=avg_confidence,
            raw_detections=all_detections
        )

    def extract_roll_number(self, gray: np.ndarray, batch_type: str = 'khoj') -> ExtractionResult:
        """Extract roll number field (3 digits)."""
        # Chayan-specific calibration
        if batch_type == 'chayan':
            origin = (1560, 1262)
            row_gap = 52
            col_gap = 65
            bubble_size = 28
        else:
            grid = self.grid_detector.detect_grid(gray, 'roll_number')
            if grid is None:
                return ExtractionResult(value="", confidence=0.0)
            origin = grid.origin
            row_gap = grid.row_gap
            col_gap = grid.col_gap
            bubble_size = grid.bubble_radius * 2

        values = FIELD_VALUES['roll_number']
        result_digits = []
        confidences = []

        for col in range(3):
            if batch_type == 'chayan':
                # Direct threshold detection for Chayan
                x = int(origin[0] + col * col_gap)
                darkest_val = 255
                darkest_row = -1

                for row in range(10):
                    y = int(origin[1] + row * row_gap)
                    if y + bubble_size > gray.shape[0] or x + bubble_size > gray.shape[1]:
                        continue
                    region = gray[y:y+bubble_size, x:x+bubble_size]
                    avg = np.mean(region)
                    if avg < darkest_val:
                        darkest_val = avg
                        darkest_row = row

                if darkest_row >= 0 and darkest_val < 180:
                    result_digits.append(values[darkest_row])
                    confidences.append(1.0 - darkest_val / 255)
                else:
                    result_digits.append('')
                    confidences.append(0.0)
            else:
                grid = self.grid_detector.detect_grid(gray, 'roll_number')
                bubble_images = self._extract_column_bubbles(gray, grid, col)

                if not bubble_images:
                    result_digits.append('')
                    confidences.append(0.0)
                    continue

                digit, conf = self.bubble_classifier.classify_and_select(bubble_images, values)
                if digit != ' ':
                    result_digits.append(digit)
                    confidences.append(conf)
                else:
                    result_digits.append('')
                    confidences.append(0.0)

        roll_number = ''.join(result_digits)
        avg_confidence = np.mean([c for c in confidences if c > 0]) if any(c > 0 for c in confidences) else 0.0

        return ExtractionResult(value=roll_number, confidence=avg_confidence)

    def extract_class(self, gray: np.ndarray, batch_type: str = 'khoj') -> ExtractionResult:
        """Extract class field (single value 3-10)."""
        values = FIELD_VALUES['class']

        if batch_type == 'chayan':
            # Chayan-specific calibration
            origin = (1775, 1266)
            row_gap = 50
            bubble_size = 28

            darkest_val = 255
            darkest_row = -1

            for row in range(8):
                x = origin[0]
                y = int(origin[1] + row * row_gap)
                if y + bubble_size > gray.shape[0] or x + bubble_size > gray.shape[1]:
                    continue
                region = gray[y:y+bubble_size, x:x+bubble_size]
                avg = np.mean(region)
                if avg < darkest_val:
                    darkest_val = avg
                    darkest_row = row

            if darkest_row >= 0 and darkest_val < 180:
                return ExtractionResult(value=values[darkest_row], confidence=1.0 - darkest_val / 255)
            return ExtractionResult(value="", confidence=0.0)
        else:
            grid = self.grid_detector.detect_grid(gray, 'class')
            if grid is None:
                return ExtractionResult(value="", confidence=0.0)

            # Single column
            bubble_images = self._extract_column_bubbles(gray, grid, 0)

            if not bubble_images:
                return ExtractionResult(value="", confidence=0.0)

            class_val, conf = self.bubble_classifier.classify_and_select(bubble_images, values)

            return ExtractionResult(value=class_val.strip(), confidence=conf)

    def extract_section(self, gray: np.ndarray, batch_type: str = 'khoj') -> ExtractionResult:
        """Extract section field (single value A-H)."""
        values = FIELD_VALUES['section']

        if batch_type == 'chayan':
            # Chayan-specific calibration
            origin = (1920, 1252)
            row_gap = 54
            bubble_size = 28

            darkest_val = 255
            darkest_row = -1

            for row in range(8):
                x = origin[0]
                y = int(origin[1] + row * row_gap)
                if y + bubble_size > gray.shape[0] or x + bubble_size > gray.shape[1]:
                    continue
                region = gray[y:y+bubble_size, x:x+bubble_size]
                avg = np.mean(region)
                if avg < darkest_val:
                    darkest_val = avg
                    darkest_row = row

            if darkest_row >= 0 and darkest_val < 180:
                return ExtractionResult(value=values[darkest_row], confidence=1.0 - darkest_val / 255)
            return ExtractionResult(value="", confidence=0.0)
        else:
            grid = self.grid_detector.detect_grid(gray, 'section')
            if grid is None:
                return ExtractionResult(value="", confidence=0.0)

            bubble_images = self._extract_column_bubbles(gray, grid, 0)

            if not bubble_images:
                return ExtractionResult(value="", confidence=0.0)

            section_val, conf = self.bubble_classifier.classify_and_select(bubble_images, values)

            return ExtractionResult(value=section_val.strip(), confidence=conf)

    def extract_school_code(self, gray: np.ndarray, batch_type: str = 'khoj') -> ExtractionResult:
        """Extract school code field (3 digits)."""
        # Chayan-specific calibration
        if batch_type == 'chayan':
            origin = (2066, 1262)
            row_gap = 52
            col_gap = 60
            bubble_size = 28
        else:
            grid = self.grid_detector.detect_grid(gray, 'school_code')
            if grid is None:
                return ExtractionResult(value="", confidence=0.0)
            origin = grid.origin
            row_gap = grid.row_gap
            col_gap = grid.col_gap
            bubble_size = grid.bubble_radius * 2

        values = FIELD_VALUES['school_code']
        result_digits = []
        confidences = []

        for col in range(3):
            if batch_type == 'chayan':
                # Direct threshold detection for Chayan
                x = int(origin[0] + col * col_gap)
                darkest_val = 255
                darkest_row = -1

                for row in range(10):
                    y = int(origin[1] + row * row_gap)
                    if y + bubble_size > gray.shape[0] or x + bubble_size > gray.shape[1]:
                        continue
                    region = gray[y:y+bubble_size, x:x+bubble_size]
                    avg = np.mean(region)
                    if avg < darkest_val:
                        darkest_val = avg
                        darkest_row = row

                if darkest_row >= 0 and darkest_val < 180:
                    result_digits.append(values[darkest_row])
                    confidences.append(1.0 - darkest_val / 255)
                else:
                    result_digits.append('')
                    confidences.append(0.0)
            else:
                grid = self.grid_detector.detect_grid(gray, 'school_code')
                bubble_images = self._extract_column_bubbles(gray, grid, col)

                if not bubble_images:
                    result_digits.append('')
                    confidences.append(0.0)
                    continue

                digit, conf = self.bubble_classifier.classify_and_select(bubble_images, values)
                if digit != ' ':
                    result_digits.append(digit)
                    confidences.append(conf)
                else:
                    result_digits.append('')
                    confidences.append(0.0)

        school_code = ''.join(result_digits)
        avg_confidence = np.mean([c for c in confidences if c > 0]) if any(c > 0 for c in confidences) else 0.0

        return ExtractionResult(value=school_code, confidence=avg_confidence)

    def extract_answers(self, gray: np.ndarray, color_image: np.ndarray = None, batch_type: str = None) -> Dict[str, str]:
        """
        Extract all 60 answers.

        Args:
            gray: Grayscale image
            color_image: Optional BGR color image for batch detection (khoj/chayan)
            batch_type: 'khoj' or 'chayan' - if None, will auto-detect from color_image

        Returns dict mapping question number (str) to answer (A/B/C/D or empty).
        """
        answers = {}
        values = FIELD_VALUES['answers']

        # Initialize all 60 questions as blank
        for q in range(1, 61):
            answers[str(q)] = ''

        # Auto-detect batch type if not provided
        if batch_type is None and color_image is not None:
            dynamic_detector = DynamicGridDetector()
            batch_type = dynamic_detector._detect_batch_type(color_image)
        elif batch_type is None:
            batch_type = 'khoj'

        # Chayan-specific calibration (exact pixspy coordinates)
        if batch_type == 'chayan':
            CHAYAN_ANSWER_GRIDS = [
                {'origin': (319, 2185), 'row_gap': 94, 'col_gap': 63},  # Block 1 (Q1-10)
                {'origin': (657, 2183), 'row_gap': 94, 'col_gap': 63},  # Block 2 (Q11-20)
                {'origin': (995, 2183), 'row_gap': 94, 'col_gap': 63},  # Block 3 (Q21-30)
                {'origin': (1332, 2183), 'row_gap': 94, 'col_gap': 63}, # Block 4 (Q31-40)
                {'origin': (1670, 2183), 'row_gap': 94, 'col_gap': 63}, # Block 5 (Q41-50)
                {'origin': (2007, 2183), 'row_gap': 94, 'col_gap': 63}, # Block 6 (Q51-60)
            ]
            bubble_size = 28

            for block_idx, block in enumerate(CHAYAN_ANSWER_GRIDS):
                start_q = block_idx * 10 + 1
                origin = block['origin']
                row_gap = block['row_gap']
                col_gap = block['col_gap']

                for row in range(10):
                    q_num = start_q + row
                    darkness_values = []

                    for col in range(4):
                        x = int(origin[0] + col * col_gap)
                        y = int(origin[1] + row * row_gap)

                        if y + bubble_size > gray.shape[0] or x + bubble_size > gray.shape[1]:
                            darkness_values.append(0)
                            continue

                        region = gray[y:y+bubble_size, x:x+bubble_size]
                        darkness = 255 - np.mean(region)
                        darkness_values.append(darkness)

                    if darkness_values:
                        max_darkness = max(darkness_values)
                        max_idx = darkness_values.index(max_darkness)
                        sorted_d = sorted(darkness_values, reverse=True)
                        gap = sorted_d[0] - sorted_d[1] if len(sorted_d) > 1 else max_darkness

                        if max_darkness >= 150:
                            answers[str(q_num)] = values[max_idx]
                        elif max_darkness >= 100 and gap >= 40:
                            answers[str(q_num)] = values[max_idx]
                        elif max_darkness >= 40 and gap >= 20:
                            answers[str(q_num)] = values[max_idx]

            return answers

        # Khoj: Use dynamic grid detection
        dynamic_detector = DynamicGridDetector()
        detected_grids = dynamic_detector.detect_answer_grids(gray, color_image)

        if not detected_grids:
            return answers

        # Process each detected block
        for block_idx, grid in enumerate(detected_grids):
            start_q = block_idx * 10 + 1

            # Create GridConfig from DetectedGrid for compatibility
            grid_config = GridConfig(
                origin=grid.origin,
                rows=grid.rows,
                cols=grid.cols,
                row_gap=grid.row_gap,
                col_gap=grid.col_gap,
                bubble_radius=grid.bubble_radius,
                confidence=grid.confidence
            )

            # Each block has 10 rows (questions) and 4 columns (A/B/C/D)
            for row in range(min(10, grid_config.rows)):
                q_num = start_q + row

                # Extract bubbles for this question
                bubble_images = self._extract_row_bubbles(gray, grid_config, row)

                if not bubble_images:
                    continue

                # Use threshold-based detection
                darkness_values = []
                for bubble_img in bubble_images:
                    if len(bubble_img.shape) == 3:
                        bubble_img = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY)
                    darkness = 255 - np.mean(bubble_img)
                    darkness_values.append(darkness)

                if darkness_values:
                    max_darkness = max(darkness_values)
                    max_idx = darkness_values.index(max_darkness)
                    sorted_d = sorted(darkness_values, reverse=True)
                    gap = sorted_d[0] - sorted_d[1] if len(sorted_d) > 1 else max_darkness

                    # Thresholds (3 tiers to handle varying fill intensities):
                    # 1. Clearly filled bubbles: darkness >= 150
                    # 2. Medium filled: darkness >= 100 with gap >= 40
                    # 3. Lightly filled: darkness >= 40 with gap >= 20 (for light markers)
                    if max_darkness >= 150:
                        answers[str(q_num)] = values[max_idx]
                    elif max_darkness >= 100 and gap >= 40:
                        answers[str(q_num)] = values[max_idx]
                    elif max_darkness >= 40 and gap >= 20:
                        answers[str(q_num)] = values[max_idx]

        return answers

    def _estimate_block_offset(
        self,
        gray: np.ndarray,
        grid: GridConfig
    ) -> Tuple[int, int]:
        """
        Estimate optimal X/Y offset for an answer block.

        Searches around the calibrated origin to find the offset that
        maximizes total bubble darkness in the first row.

        Args:
            gray: Grayscale image
            grid: Grid configuration for this block

        Returns:
            (x_offset, y_offset) to adjust the grid origin
        """
        origin_x, origin_y = grid.origin
        col_gap = grid.col_gap

        best_x_offset = 0
        best_y_offset = 0
        best_total_darkness = 0

        # Sample darkness at different offsets
        def sample_darkness(x, y):
            x, y = int(x), int(y)
            r = 14
            x1, y1 = max(0, x - r), max(0, y - r)
            x2, y2 = min(gray.shape[1], x + r), min(gray.shape[0], y + r)
            if x2 <= x1 or y2 <= y1:
                return 0
            return 255 - np.mean(gray[y1:y2, x1:x2])

        # Search in a grid around the calibrated position
        for y_off in range(-20, 21, 5):
            for x_off in range(-20, 21, 5):
                total_darkness = 0

                # Sample all 4 options in first row
                for col in range(4):
                    x = origin_x + x_off + col * col_gap
                    y = origin_y + y_off
                    total_darkness += sample_darkness(x, y)

                if total_darkness > best_total_darkness:
                    best_total_darkness = total_darkness
                    best_x_offset = x_off
                    best_y_offset = y_off

        return (best_x_offset, best_y_offset)

    def _extract_column_bubbles(
        self,
        gray: np.ndarray,
        grid: GridConfig,
        col: int,
        extra_x: int = 0
    ) -> List[np.ndarray]:
        """Extract bubble images for all rows in a column."""
        bubbles = []
        h, w = gray.shape[:2]

        for row in range(grid.rows):
            cx, cy = self.grid_detector.get_bubble_position(grid, row, col)
            cx += extra_x  # Apply extra X offset for Chayan name sections

            # Calculate bounding box
            r = grid.bubble_radius + 2  # Small padding
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)

            if x2 > x1 and y2 > y1:
                bubble = gray[y1:y2, x1:x2]
                if bubble.size > 0:
                    bubbles.append(bubble)
                else:
                    # Empty placeholder
                    bubbles.append(np.full((28, 28), 255, dtype=np.uint8))
            else:
                bubbles.append(np.full((28, 28), 255, dtype=np.uint8))

        return bubbles

    def _extract_row_bubbles(
        self,
        gray: np.ndarray,
        grid: GridConfig,
        row: int
    ) -> List[np.ndarray]:
        """Extract bubble images for all columns in a row (for answer grids)."""
        bubbles = []
        h, w = gray.shape[:2]

        for col in range(grid.cols):
            cx, cy = self.grid_detector.get_bubble_position(grid, row, col)

            # Use fixed radius=14 for consistency with offset estimation
            # This matches the sampling radius used in _estimate_block_offset
            r = 14
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)

            if x2 > x1 and y2 > y1:
                bubble = gray[y1:y2, x1:x2]
                if bubble.size > 0:
                    bubbles.append(bubble)
                else:
                    bubbles.append(np.full((28, 28), 255, dtype=np.uint8))
            else:
                bubbles.append(np.full((28, 28), 255, dtype=np.uint8))

        return bubbles

    def _clean_name(self, raw_name: str) -> str:
        """
        Clean extracted name by removing trailing spaces and fixing common issues.

        Args:
            raw_name: Raw extracted name with potential spaces and artifacts

        Returns:
            Cleaned name string
        """
        # Strip leading/trailing whitespace
        name = raw_name.strip()

        # Collapse multiple spaces to single space
        import re
        name = re.sub(r'\s+', ' ', name)

        # Remove trailing single characters (often artifacts)
        words = name.split()
        cleaned_words = []
        for word in words:
            # Keep words of length >= 2, or common single letters
            if len(word) >= 2 or word in ['A', 'I']:
                cleaned_words.append(word)

        return ' '.join(cleaned_words)

    def _calculate_answers_confidence(self, answers: Dict[str, str]) -> float:
        """Calculate overall confidence for answers extraction (0-100 scale)."""
        # Simple metric: percentage of questions with answers
        answered = sum(1 for ans in answers.values() if ans)
        total = len(answers)
        # Return as percentage (0-100) not fraction (0-1)
        return (answered / total * 100) if total > 0 else 0.0


def extract_all_fields(gray: np.ndarray) -> OMRExtractionResult:
    """
    Convenience function to extract all fields from an aligned image.

    Args:
        gray: Grayscale image aligned to canonical frame

    Returns:
        OMRExtractionResult with all extracted values
    """
    extractor = FieldExtractor()
    return extractor.extract_all_fields(gray)

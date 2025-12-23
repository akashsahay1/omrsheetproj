"""
Main OMR Processing Engine
"""
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from src.core.template import Template, TemplateLoader, FieldBlock
from src.core.bubble_detector import BubbleDetector
from src.core.image_processor import ImageProcessor
from src.core.alignment import AlignmentDetector
from src.models.candidate import Candidate
from src.models.result import ProcessingResult
from src.config.constants import SheetConfig, MULTI_MARK_VALUE, EMPTY_VALUE
from src.utils.image_utils import ImageUtils
from src.utils.logging_utils import get_logger

logger = get_logger('omr_engine')


class OMREngine:
    """
    Main OMR processing engine for Vidya Bharti answer sheets

    Handles:
    - Image loading and preprocessing
    - Sheet alignment and correction
    - Bubble detection for all fields
    - Data extraction (name, roll number, answers, etc.)
    """

    def __init__(self, template: Optional[Template] = None, template_name: str = "vidya_bharti_2025"):
        """
        Initialize OMR engine

        Args:
            template: Template configuration (uses default if not provided)
            template_name: Name of template folder to load from
        """
        self.template_loader = TemplateLoader()

        if template is None:
            # Try to load from JSON file first
            loaded_template = self.template_loader.load_template(template_name)
            if loaded_template is not None:
                self.template = loaded_template
                logger.info(f"Loaded template from JSON: {template_name}")
            else:
                # Fall back to hardcoded default
                self.template = self.template_loader.create_default_template()
                logger.warning("Using hardcoded default template")
        else:
            self.template = template

        # Initialize components
        self.image_processor = ImageProcessor(
            target_width=self.template.page_dimensions[0],
            target_height=self.template.page_dimensions[1]
        )
        self.bubble_detector = BubbleDetector()
        self.alignment_detector = AlignmentDetector()

        # Processing state
        self._current_image: Optional[np.ndarray] = None
        self._processed_image: Optional[np.ndarray] = None

    def load_template(self, template_name: str) -> bool:
        """
        Load template by name

        Args:
            template_name: Template folder name

        Returns:
            True if loaded successfully
        """
        template = self.template_loader.load_template(template_name)

        if template is not None:
            self.template = template
            self.image_processor = ImageProcessor(
                target_width=template.page_dimensions[0],
                target_height=template.page_dimensions[1]
            )
            return True

        return False

    def process_image(self, image_path: str) -> ProcessingResult:
        """
        Process a single OMR sheet image

        Args:
            image_path: Path to image file

        Returns:
            ProcessingResult with extracted data
        """
        start_time = time.time()
        result = ProcessingResult(source_file=image_path)

        try:
            # Load image
            image = ImageUtils.load_image(image_path)
            if image is None:
                result.error_message = "Failed to load image"
                return result

            self._current_image = image

            # Preprocess
            processed, resized = self.image_processor.preprocess(image)
            self._processed_image = processed

            # Auto-align if needed
            processed, rotation = self.alignment_detector.auto_rotate(processed)

            # Extract candidate data
            candidate = self._extract_all_fields(processed)
            candidate.source_image = image_path

            # Build result
            result.success = True
            result.candidate = candidate
            result.alignment_confidence = self.alignment_detector.confidence
            result.processing_time_ms = (time.time() - start_time) * 1000

            # Check for issues
            if candidate.multi_marked_count > 0:
                result.warnings.append(
                    f"{candidate.multi_marked_count} questions have multiple marks"
                )

            if candidate.unanswered_count > 30:
                result.warnings.append("More than half questions unanswered")
                result.needs_manual_review = True

            if not candidate.name:
                result.warnings.append("Name could not be extracted")

            logger.info(f"Processed {image_path}: name='{candidate.name}', "
                       f"roll='{candidate.roll_number}'")

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Error processing {image_path}: {e}")

        return result

    def _extract_all_fields(self, image: np.ndarray) -> Candidate:
        """
        Extract all fields from preprocessed image

        Args:
            image: Preprocessed grayscale image

        Returns:
            Candidate with extracted data
        """
        candidate = Candidate()

        # Extract name
        candidate.name = self._extract_name(image)

        # Extract roll number
        candidate.roll_number = self._extract_roll_number(image)

        # Extract class
        candidate.class_ = self._extract_class(image)

        # Extract section
        candidate.section = self._extract_section(image)

        # Extract school code
        candidate.school_code = self._extract_school_code(image)

        # Extract answers
        answers = self._extract_answers(image)
        for q_num, answer in answers.items():
            candidate.set_answer(q_num, answer)

        return candidate

    def _extract_name(self, image: np.ndarray) -> str:
        """Extract candidate name from bubble grid"""
        field_block = self.template.get_field_block('CandidateName')

        if field_block is None:
            logger.warning("CandidateName field block not found in template")
            return ""

        # Get bubble coordinates
        coords = field_block.get_bubble_coordinates()

        # Detect marked values
        values = self.bubble_detector.detect_marked_value(image, coords)

        # Build name from detected characters
        name_chars = []
        for i in range(1, SheetConfig.NAME_COLUMNS + 1):
            label = f'Name_{i}'
            char = values.get(label, '')

            if char and char != MULTI_MARK_VALUE:
                name_chars.append(char)
            elif not char:
                # Stop at first empty position (end of name)
                # But allow spaces in the middle
                if name_chars and all(c == '' for c in name_chars[-3:] if c == ''):
                    break

        # Join and clean name
        name = ''.join(name_chars).strip()

        logger.debug(f"Extracted name: '{name}'")
        return name

    def _extract_roll_number(self, image: np.ndarray) -> str:
        """Extract 3-digit roll number"""
        field_block = self.template.get_field_block('RollNumber')

        if field_block is None:
            logger.warning("RollNumber field block not found")
            return ""

        coords = field_block.get_bubble_coordinates()
        values = self.bubble_detector.detect_marked_value(image, coords)

        # Build roll number
        digits = []
        for i, label in enumerate(['Roll_D1', 'Roll_D2', 'Roll_D3']):
            digit = values.get(label, '')
            if digit and digit != MULTI_MARK_VALUE:
                digits.append(digit)
            else:
                digits.append('0')  # Default to 0 if not detected

        roll_number = ''.join(digits)

        logger.debug(f"Extracted roll number: '{roll_number}'")
        return roll_number

    def _extract_class(self, image: np.ndarray) -> str:
        """Extract class (0-9)"""
        field_block = self.template.get_field_block('Class')

        if field_block is None:
            return ""

        coords = field_block.get_bubble_coordinates()
        values = self.bubble_detector.detect_marked_value(image, coords)

        class_value = values.get('Class', '')

        logger.debug(f"Extracted class: '{class_value}'")
        return class_value

    def _extract_section(self, image: np.ndarray) -> str:
        """Extract section (A-H)"""
        field_block = self.template.get_field_block('Section')

        if field_block is None:
            return ""

        coords = field_block.get_bubble_coordinates()
        values = self.bubble_detector.detect_marked_value(image, coords)

        section = values.get('Section', '')

        logger.debug(f"Extracted section: '{section}'")
        return section

    def _extract_school_code(self, image: np.ndarray) -> str:
        """Extract 4-digit school code"""
        field_block = self.template.get_field_block('SchoolCode')

        if field_block is None:
            return ""

        coords = field_block.get_bubble_coordinates()
        values = self.bubble_detector.detect_marked_value(image, coords)

        # Build school code
        digits = []
        for i, label in enumerate(['School_D1', 'School_D2', 'School_D3', 'School_D4']):
            digit = values.get(label, '')
            if digit and digit != MULTI_MARK_VALUE:
                digits.append(digit)
            else:
                digits.append('0')

        school_code = ''.join(digits)

        logger.debug(f"Extracted school code: '{school_code}'")
        return school_code

    def _extract_answers(self, image: np.ndarray) -> Dict[int, str]:
        """
        Extract answers for all 60 questions

        Returns:
            Dict mapping question number (1-60) to answer (A/B/C/D/MULTI/empty)
        """
        answers = {}

        # Process each answer block
        for col in range(6):
            start_q = col * 10 + 1
            end_q = start_q + 9
            block_name = f'Answers_{start_q}_{end_q}'

            field_block = self.template.get_field_block(block_name)

            if field_block is None:
                logger.warning(f"Answer block {block_name} not found")
                # Fill with empty values
                for q in range(start_q, end_q + 1):
                    answers[q] = ""
                continue

            coords = field_block.get_bubble_coordinates()
            values = self.bubble_detector.detect_marked_value(image, coords)

            # Map to question numbers
            for q_num in range(start_q, end_q + 1):
                label = f'q{q_num}'
                answer = values.get(label, '')
                answers[q_num] = answer

        logger.debug(f"Extracted {len(answers)} answers")
        return answers

    def calibrate_with_filled_sheet(self,
                                     filled_image_path: str,
                                     known_values: Dict[str, Any]) -> bool:
        """
        Calibrate detection thresholds using a known filled sheet

        Args:
            filled_image_path: Path to filled sheet image
            known_values: Dict with known correct values for calibration

        Returns:
            True if calibration successful
        """
        try:
            image = ImageUtils.load_image(filled_image_path)
            if image is None:
                return False

            processed, _ = self.image_processor.preprocess(image)

            # Collect filled and empty bubble samples
            filled_coords = []
            empty_coords = []

            # Use known answers to identify filled/empty bubbles
            if 'answers' in known_values:
                known_answers = known_values['answers']

                for col in range(6):
                    start_q = col * 10 + 1
                    block_name = f'Answers_{start_q}_{start_q + 9}'
                    field_block = self.template.get_field_block(block_name)

                    if field_block is None:
                        continue

                    coords = field_block.get_bubble_coordinates()

                    for coord in coords:
                        q_num = int(coord['label'].replace('q', ''))

                        if q_num in known_answers:
                            correct_answer = known_answers[q_num]

                            if coord['value'] == correct_answer:
                                filled_coords.append(coord)
                            else:
                                empty_coords.append(coord)

            # Calibrate threshold
            if filled_coords and empty_coords:
                optimal_threshold = self.bubble_detector.calibrate_threshold(
                    processed, filled_coords, empty_coords
                )
                self.bubble_detector.set_threshold(optimal_threshold)

                logger.info(f"Calibration complete. New threshold: {optimal_threshold}")
                return True

            return False

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return False

    def set_fill_threshold(self, threshold: int) -> None:
        """Set bubble fill detection threshold"""
        self.bubble_detector.set_threshold(threshold)

    def get_debug_image(self) -> Optional[np.ndarray]:
        """Get processed image for debugging"""
        return self._processed_image

    def process_image_with_debug(self, image_path: str) -> Tuple[ProcessingResult, Dict[str, np.ndarray]]:
        """
        Process image and return debug visualizations

        Args:
            image_path: Path to image

        Returns:
            Tuple of (result, debug_images_dict)
        """
        result = self.process_image(image_path)

        debug_images = {}

        if self._processed_image is not None:
            debug_images['processed'] = self._processed_image.copy()

            # Create visualization with detected bubbles
            if result.candidate:
                vis_image = cv2.cvtColor(self._processed_image, cv2.COLOR_GRAY2BGR)
                debug_images['visualization'] = vis_image

        return result, debug_images

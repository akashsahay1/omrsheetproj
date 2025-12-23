"""
Data validation service
"""
from typing import List, Dict, Any, Optional
from src.models.candidate import Candidate
from src.models.answer_key import AnswerKey
from src.config.constants import SheetConfig
from src.utils.logging_utils import get_logger

logger = get_logger('validator')


class ValidationResult:
    """Result of validation"""

    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, message: str) -> None:
        """Add error and mark as invalid"""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add warning (doesn't affect validity)"""
        self.warnings.append(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings
        }


class Validator:
    """
    Validates extracted data and configurations
    """

    @staticmethod
    def validate_candidate(candidate: Candidate) -> ValidationResult:
        """
        Validate extracted candidate data

        Args:
            candidate: Candidate to validate

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()

        # Validate name
        if not candidate.name:
            result.add_warning("Name is empty")
        elif len(candidate.name) > SheetConfig.NAME_COLUMNS:
            result.add_warning(f"Name exceeds {SheetConfig.NAME_COLUMNS} characters")
        elif not candidate.name.replace(' ', '').isalpha():
            result.add_warning("Name contains non-alphabetic characters")

        # Validate roll number
        if not candidate.roll_number:
            result.add_warning("Roll number is empty")
        elif len(candidate.roll_number) != SheetConfig.ROLL_NUMBER_DIGITS:
            result.add_warning(f"Roll number should be {SheetConfig.ROLL_NUMBER_DIGITS} digits")
        elif not candidate.roll_number.isdigit():
            result.add_error("Roll number contains non-numeric characters")

        # Validate class
        if candidate.class_:
            if candidate.class_ not in SheetConfig.CLASS_OPTIONS:
                result.add_error(f"Invalid class value: {candidate.class_}")

        # Validate section
        if candidate.section:
            if candidate.section not in SheetConfig.SECTION_OPTIONS:
                result.add_error(f"Invalid section value: {candidate.section}")

        # Validate school code
        if not candidate.school_code:
            result.add_warning("School code is empty")
        elif len(candidate.school_code) != SheetConfig.SCHOOL_CODE_DIGITS:
            result.add_warning(f"School code should be {SheetConfig.SCHOOL_CODE_DIGITS} digits")
        elif not candidate.school_code.isdigit():
            result.add_error("School code contains non-numeric characters")

        # Validate answers
        invalid_answers = []
        for q_num, answer in candidate.answers.items():
            if answer and answer not in SheetConfig.ANSWER_OPTIONS and answer != "MULTI":
                invalid_answers.append(q_num)

        if invalid_answers:
            result.add_error(f"Invalid answer values for questions: {invalid_answers}")

        # Check for multi-marks
        if candidate.multi_marked_count > 0:
            result.add_warning(
                f"{candidate.multi_marked_count} questions have multiple marks"
            )

        # Check for too many unanswered
        if candidate.unanswered_count > 30:
            result.add_warning("More than half of questions are unanswered")

        return result

    @staticmethod
    def validate_answer_key(answer_key: AnswerKey) -> ValidationResult:
        """
        Validate answer key

        Args:
            answer_key: Answer key to validate

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        # Check completeness
        if not answer_key.is_complete():
            missing = 60 - answer_key.get_filled_count()
            result.add_error(f"Answer key is incomplete: {missing} answers missing")

        # Check for invalid values
        invalid = []
        for q_num, answer in answer_key.answers.items():
            if answer and answer not in SheetConfig.ANSWER_OPTIONS:
                invalid.append((q_num, answer))

        if invalid:
            result.add_error(f"Invalid answers in key: {invalid}")

        return result

    @staticmethod
    def validate_batch_input(folder_path: str,
                             image_files: List[str]) -> ValidationResult:
        """
        Validate batch processing input

        Args:
            folder_path: Input folder path
            image_files: List of image files found

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        if not folder_path:
            result.add_error("Input folder path is empty")
            return result

        if not image_files:
            result.add_error("No image files found in folder")
            return result

        if len(image_files) > 1000:
            result.add_warning(f"Large batch: {len(image_files)} files. Processing may take a while.")

        return result

    @staticmethod
    def validate_export_path(output_path: str) -> ValidationResult:
        """
        Validate export output path

        Args:
            output_path: Proposed output path

        Returns:
            ValidationResult
        """
        result = ValidationResult()

        if not output_path:
            result.add_error("Output path is empty")
            return result

        if not output_path.endswith('.xlsx'):
            result.add_warning("Output file should have .xlsx extension")

        return result

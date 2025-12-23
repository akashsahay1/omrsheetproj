"""
OMRChecker Integration Engine

Directly imports and uses OMRChecker library for processing.
"""
import os
import sys
import shutil
import csv
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.models.candidate import Candidate
from src.models.result import ProcessingResult
from src.utils.logging_utils import get_logger

logger = get_logger('omrchecker_engine')


def get_base_path() -> Path:
    """Get base path - works for both dev and bundled app."""
    if getattr(sys, 'frozen', False):
        # Running in a bundle (PyInstaller)
        return Path(sys._MEIPASS)
    else:
        # Running in development
        return Path(__file__).parent.parent.parent


def get_writable_path() -> Path:
    """Get a writable path for temp files and outputs."""
    if getattr(sys, 'frozen', False):
        # In bundled app, use user's temp directory
        app_temp = Path(tempfile.gettempdir()) / 'VBOMRScanner'
        app_temp.mkdir(exist_ok=True)
        return app_temp
    else:
        # Development mode - use project directory
        return Path(__file__).parent.parent.parent


# Get project paths
PROJECT_ROOT = get_base_path()
OMRCHECKER_PATH = PROJECT_ROOT / 'OMRChecker'
WRITABLE_PATH = get_writable_path()

# Add OMRChecker to path for imports
if str(OMRCHECKER_PATH) not in sys.path:
    sys.path.insert(0, str(OMRCHECKER_PATH))


class OMRCheckerEngine:
    """
    OMR processing engine using OMRChecker library directly.
    """

    DEFAULT_TEMPLATE_PATH = OMRCHECKER_PATH / 'inputs' / 'VidyaBharti2025' / 'template.json'

    def __init__(self, template_path: Optional[str] = None):
        """
        Initialize OMRChecker engine.

        Args:
            template_path: Path to template.json file
        """
        self.template_path = Path(template_path) if template_path else self.DEFAULT_TEMPLATE_PATH
        self._temp_dir = None
        # Use writable path for outputs
        self._outputs_dir = WRITABLE_PATH / 'outputs'
        self._outputs_dir.mkdir(exist_ok=True)

    def process_image(self, image_path: str) -> ProcessingResult:
        """
        Process a single OMR image using OMRChecker.

        Args:
            image_path: Path to the OMR image

        Returns:
            ProcessingResult with extracted candidate data
        """
        start_time = time.time()

        try:
            # Create temporary directory for this image (use writable path)
            self._temp_dir = WRITABLE_PATH / 'inputs' / '_temp_processing'
            self._temp_dir.mkdir(parents=True, exist_ok=True)

            # Copy image to temp directory
            image_name = Path(image_path).name
            temp_image = self._temp_dir / image_name
            shutil.copy2(image_path, temp_image)

            # Copy template to temp directory
            if self.template_path.exists():
                shutil.copy2(self.template_path, self._temp_dir / 'template.json')

                # Also copy config.json if it exists
                config_path = self.template_path.parent / 'config.json'
                if config_path.exists():
                    shutil.copy2(config_path, self._temp_dir / 'config.json')
            else:
                return ProcessingResult(
                    success=False,
                    error_message=f"Template not found: {self.template_path}",
                    processing_time_ms=(time.time() - start_time) * 1000
                )

            # Run OMRChecker
            result_data = self._run_omrchecker()

            # Clean up temp directory
            self._cleanup()

            if result_data:
                candidate = self._convert_to_candidate(result_data, image_path)
                processing_time = (time.time() - start_time) * 1000

                return ProcessingResult(
                    success=True,
                    candidate=candidate,
                    source_file=image_path,
                    processing_time_ms=processing_time
                )
            else:
                return ProcessingResult(
                    success=False,
                    error_message="OMRChecker returned no results",
                    processing_time_ms=(time.time() - start_time) * 1000
                )

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._cleanup()
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def _run_omrchecker(self) -> Optional[Dict[str, Any]]:
        """Run OMRChecker directly and parse results."""
        try:
            # Save original sys.path and sys.modules state
            original_path = sys.path.copy()
            original_modules = {k: v for k, v in sys.modules.items() if k.startswith('src')}

            # Remove our app's src modules temporarily
            for key in list(sys.modules.keys()):
                if key.startswith('src') and 'omr_processor' not in key:
                    del sys.modules[key]

            # Add OMRChecker path at the front
            sys.path.insert(0, str(OMRCHECKER_PATH))

            try:
                # Now import OMRChecker's entry point
                from src.entry import entry_point

                # Prepare arguments
                args = {
                    'debug': False,
                    'output_dir': str(self._outputs_dir),
                    'autoAlign': False,
                    'setLayout': False
                }

                logger.info(f"Running OMRChecker on: {self._temp_dir}")

                # Run OMRChecker
                entry_point(self._temp_dir, args)
            finally:
                # Restore original state
                sys.path = original_path
                # Remove OMRChecker's src modules
                for key in list(sys.modules.keys()):
                    if key.startswith('src') and key not in original_modules:
                        del sys.modules[key]
                # Restore our app's src modules
                sys.modules.update(original_modules)

            # Parse the results from CSV
            results_dir = self._outputs_dir / 'Results'
            csv_files = list(results_dir.glob('Results_*.csv'))

            if csv_files:
                # Get the most recent CSV file
                csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Reading results from: {csv_file}")

                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    result_row = None
                    for row in reader:
                        # Get the last row (most recent result)
                        result_row = dict(row)

                    return result_row

            logger.warning("No CSV results found")
            return None

        except Exception as e:
            logger.error(f"Error running OMRChecker: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _cleanup(self):
        """Clean up temporary files."""
        try:
            if self._temp_dir and self._temp_dir.exists():
                for f in self._temp_dir.iterdir():
                    if f.is_file():
                        f.unlink()
                self._temp_dir.rmdir()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

    def _convert_to_candidate(self, result_data: Dict[str, Any], source_image: str) -> Candidate:
        """Convert OMRChecker CSV row to Candidate model."""
        candidate = Candidate(source_image=source_image)

        # Extract Candidate Name (name1..name25)
        has_individual_cols = any(f'name{i}' in result_data for i in range(1, 26))

        if has_individual_cols:
            # Build name from individual columns, treating empty columns as spaces
            name_chars = []
            last_non_empty = 0

            for i in range(1, 26):
                key = f'name{i}'
                if key in result_data:
                    char = str(result_data[key]).strip() if result_data[key] else ''
                    if char:
                        # Add spaces for any skipped empty columns
                        while len(name_chars) < i - 1:
                            name_chars.append(' ')
                        name_chars.append(char)
                        last_non_empty = len(name_chars)

            # Trim trailing spaces but keep internal ones
            candidate.name = ''.join(name_chars[:last_non_empty]).strip()
        elif 'CandidateName' in result_data:
            candidate.name = str(result_data['CandidateName']).strip()
        else:
            candidate.name = ''

        # Extract Roll Number
        if 'Roll' in result_data:
            candidate.roll_number = str(result_data['Roll']).strip()
        else:
            # Try individual roll columns
            roll_digits = []
            for i in range(1, 4):
                key = f'roll{i}'
                if key in result_data and result_data[key]:
                    roll_digits.append(str(result_data[key]))
            candidate.roll_number = ''.join(roll_digits).strip()

        # Extract Class
        if 'class' in result_data and result_data['class']:
            candidate.class_ = str(result_data['class'])

        # Extract Section
        if 'section' in result_data and result_data['section']:
            candidate.section = str(result_data['section'])

        # Extract School Code
        if 'SchoolCode' in result_data:
            candidate.school_code = str(result_data['SchoolCode']).strip()
        else:
            school_digits = []
            for i in range(1, 5):
                key = f'school{i}'
                if key in result_data and result_data[key]:
                    school_digits.append(str(result_data[key]))
            candidate.school_code = ''.join(school_digits).strip()

        # Extract Answers (q1-q60)
        multi_marked = []
        unanswered = []

        for q in range(1, 61):
            key = f'q{q}'
            if key in result_data:
                answer = result_data[key]
                if answer:
                    answer = str(answer).upper().strip()
                    if len(answer) > 1:
                        multi_marked.append(q)
                        candidate.answers[q] = 'MULTI'
                    else:
                        candidate.answers[q] = answer
                else:
                    unanswered.append(q)
                    candidate.answers[q] = ''
            else:
                unanswered.append(q)
                candidate.answers[q] = ''

        candidate.multi_marked_questions = multi_marked
        candidate.unanswered_questions = unanswered

        return candidate


def process_omr_image(image_path: str, template_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Process an OMR image and return results.
    """
    engine = OMRCheckerEngine(template_path)
    result = engine.process_image(image_path)

    if result.success and result.candidate:
        return {
            'success': True,
            'name': result.candidate.name,
            'roll_number': result.candidate.roll_number,
            'class': result.candidate.class_,
            'section': result.candidate.section,
            'school_code': result.candidate.school_code,
            'answers': result.candidate.answers,
            'multi_marked': result.candidate.multi_marked_questions,
            'unanswered': result.candidate.unanswered_questions
        }
    else:
        return {
            'success': False,
            'error': result.error_message
        }

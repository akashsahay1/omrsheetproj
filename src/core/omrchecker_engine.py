"""
OMRChecker Integration Engine

Directly imports and uses OMRChecker library for processing.
Optimized for batch processing with one-time module loading.
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


# =============================================================================
# Global OMRChecker Module Management - ONE-TIME IMPORT for performance
# =============================================================================

_omrchecker_modules_loaded = False
_omrchecker_entry_point = None
_original_sys_path = None
_original_src_modules = None


def _setup_omrchecker_imports():
    """
    One-time setup of OMRChecker imports.
    Imports OMRChecker modules and keeps them loaded for all processing.
    """
    global _omrchecker_modules_loaded, _omrchecker_entry_point
    global _original_sys_path, _original_src_modules

    if _omrchecker_modules_loaded:
        return True

    try:
        # Save original state
        _original_sys_path = sys.path.copy()
        _original_src_modules = {k: v for k, v in sys.modules.items() if k.startswith('src')}

        # Temporarily remove our app's src modules
        for key in list(sys.modules.keys()):
            if key.startswith('src') and 'omr_processor' not in key:
                del sys.modules[key]

        # Add OMRChecker path at the front
        sys.path.insert(0, str(OMRCHECKER_PATH))

        # Import OMRChecker's entry point
        from src.entry import entry_point
        _omrchecker_entry_point = entry_point

        _omrchecker_modules_loaded = True
        logger.info("OMRChecker modules loaded successfully (one-time)")
        return True

    except Exception as e:
        logger.error(f"Failed to load OMRChecker modules: {e}")
        # Restore original state on failure
        if _original_sys_path:
            sys.path = _original_sys_path
        if _original_src_modules:
            sys.modules.update(_original_src_modules)
        return False


def _restore_app_modules():
    """Restore app modules after OMRChecker processing."""
    global _original_sys_path, _original_src_modules

    if _original_sys_path:
        sys.path = _original_sys_path
    if _original_src_modules:
        # Remove OMRChecker's src modules that conflict
        for key in list(sys.modules.keys()):
            if key.startswith('src') and key not in _original_src_modules:
                del sys.modules[key]
        # Restore our app's src modules
        sys.modules.update(_original_src_modules)


class OMRCheckerEngine:
    """
    OMR processing engine using OMRChecker library directly.

    Optimized for batch processing:
    - One-time module import (saved ~3-5s per image)
    - Batch directory setup (template copied once)
    - Minimal file operations per image
    """

    DEFAULT_TEMPLATE_PATH = OMRCHECKER_PATH / 'inputs' / 'VidyaBharti2025' / 'template.json'

    # Class-level batch state for optimization
    _batch_dir = None
    _batch_initialized = False

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

        # Initialize OMRChecker modules ONCE
        _setup_omrchecker_imports()

    def initialize_batch(self) -> bool:
        """
        Initialize batch processing directory.
        Call once before processing multiple images.
        Sets up template and config once for the entire batch.
        """
        if OMRCheckerEngine._batch_initialized:
            return True

        try:
            # Create batch processing directory
            OMRCheckerEngine._batch_dir = WRITABLE_PATH / 'inputs' / '_batch_processing'
            OMRCheckerEngine._batch_dir.mkdir(parents=True, exist_ok=True)

            # Copy template once
            if self.template_path.exists():
                shutil.copy2(self.template_path, OMRCheckerEngine._batch_dir / 'template.json')

                # Copy config.json if exists
                config_path = self.template_path.parent / 'config.json'
                if config_path.exists():
                    shutil.copy2(config_path, OMRCheckerEngine._batch_dir / 'config.json')

                OMRCheckerEngine._batch_initialized = True
                logger.info("Batch processing initialized (template copied once)")
                return True
            else:
                logger.error(f"Template not found: {self.template_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize batch: {e}")
            return False

    def finalize_batch(self):
        """Clean up batch processing directory."""
        try:
            if OMRCheckerEngine._batch_dir and OMRCheckerEngine._batch_dir.exists():
                # Remove all files in batch dir
                for f in OMRCheckerEngine._batch_dir.iterdir():
                    if f.is_file():
                        f.unlink()
                OMRCheckerEngine._batch_dir.rmdir()

            OMRCheckerEngine._batch_initialized = False
            OMRCheckerEngine._batch_dir = None
            logger.info("Batch processing finalized")

        except Exception as e:
            logger.warning(f"Batch cleanup error: {e}")

    def process_image(self, image_path: str) -> ProcessingResult:
        """
        Process a single OMR image using OMRChecker.

        Optimized version:
        - Uses pre-initialized batch directory
        - Minimal file copying (just the image)
        - Reuses loaded OMRChecker modules

        Args:
            image_path: Path to the OMR image

        Returns:
            ProcessingResult with extracted candidate data
        """
        start_time = time.time()

        try:
            # Ensure batch is initialized (auto-init if needed)
            if not OMRCheckerEngine._batch_initialized:
                if not self.initialize_batch():
                    return ProcessingResult(
                        success=False,
                        error_message=f"Template not found: {self.template_path}",
                        processing_time_ms=(time.time() - start_time) * 1000
                    )

            # Use batch directory (template already there)
            self._temp_dir = OMRCheckerEngine._batch_dir

            # Copy ONLY the image (template already in place)
            image_name = Path(image_path).name
            temp_image = self._temp_dir / image_name
            shutil.copy2(image_path, temp_image)

            # Run OMRChecker (using pre-loaded modules)
            result_data = self._run_omrchecker_optimized()

            # Remove just the processed image (keep template)
            if temp_image.exists():
                temp_image.unlink()

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
            return ProcessingResult(
                success=False,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def _run_omrchecker_optimized(self) -> Optional[Dict[str, Any]]:
        """
        Run OMRChecker using pre-loaded modules (OPTIMIZED).
        Skips module import/unload cycle - saves ~3-5 seconds per image.
        """
        global _omrchecker_entry_point

        try:
            if not _omrchecker_modules_loaded or _omrchecker_entry_point is None:
                logger.error("OMRChecker modules not loaded!")
                return self._run_omrchecker_fallback()

            # Prepare arguments
            args = {
                'debug': False,
                'output_dir': str(self._outputs_dir),
                'autoAlign': False,
                'setLayout': False
            }

            # Run OMRChecker using pre-loaded entry point
            _omrchecker_entry_point(self._temp_dir, args)

            # Parse the results from CSV
            return self._parse_results()

        except Exception as e:
            logger.error(f"Error running OMRChecker (optimized): {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fall back to traditional method
            return self._run_omrchecker_fallback()

    def _run_omrchecker_fallback(self) -> Optional[Dict[str, Any]]:
        """Fallback method - reloads modules each time (slower but reliable)."""
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
                # Import OMRChecker's entry point
                from src.entry import entry_point

                args = {
                    'debug': False,
                    'output_dir': str(self._outputs_dir),
                    'autoAlign': False,
                    'setLayout': False
                }

                logger.info(f"Running OMRChecker (fallback) on: {self._temp_dir}")
                entry_point(self._temp_dir, args)

            finally:
                # Restore original state
                sys.path = original_path
                for key in list(sys.modules.keys()):
                    if key.startswith('src') and key not in original_modules:
                        del sys.modules[key]
                sys.modules.update(original_modules)

            return self._parse_results()

        except Exception as e:
            logger.error(f"Error running OMRChecker (fallback): {e}")
            return None

    def _parse_results(self) -> Optional[Dict[str, Any]]:
        """Parse OMRChecker CSV results."""
        try:
            results_dir = self._outputs_dir / 'Results'
            csv_files = list(results_dir.glob('Results_*.csv'))

            if csv_files:
                csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)

                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    result_row = None
                    for row in reader:
                        result_row = dict(row)
                    return result_row

            logger.warning("No CSV results found")
            return None

        except Exception as e:
            logger.error(f"Error parsing results: {e}")
            return None

    def _cleanup(self):
        """Clean up temporary files (legacy method)."""
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

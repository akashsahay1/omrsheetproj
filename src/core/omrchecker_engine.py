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
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np

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
    _template_dimensions = None  # (width, height) from template

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

        # Validate template exists
        self._validate_template()

        # Load template dimensions for image normalization
        self._load_template_dimensions()

        # Initialize OMRChecker modules ONCE
        _setup_omrchecker_imports()

    def _load_template_dimensions(self) -> None:
        """Load page dimensions from template.json."""
        try:
            with open(self.template_path, 'r') as f:
                template = json.load(f)

            if 'pageDimensions' in template:
                dims = template['pageDimensions']
                OMRCheckerEngine._template_dimensions = (dims[0], dims[1])
                logger.info(f"Template dimensions: {OMRCheckerEngine._template_dimensions}")
            else:
                logger.warning("No dimensions in template")
                OMRCheckerEngine._template_dimensions = None
        except Exception as e:
            logger.error(f"Failed to load template dimensions: {e}")
            OMRCheckerEngine._template_dimensions = None

    def _copy_template_files(self, dest_dir: Path) -> None:
        """
        Copy template files (template.json and reference image) to destination directory.

        FeatureBasedAlignment handles image dimension normalization automatically,
        so we just copy the template files as-is without any coordinate scaling.
        """
        # Copy template.json
        shutil.copy2(self.template_path, dest_dir / 'template.json')

        # Copy reference image for FeatureBasedAlignment (omr-filled.png)
        reference_image = self.template_path.parent / 'omr-filled.png'
        if reference_image.exists():
            shutil.copy2(reference_image, dest_dir / 'omr-filled.png')
            logger.debug(f"Copied reference image for alignment: {reference_image.name}")

        # Copy config.json if exists
        config_path = self.template_path.parent / 'config.json'
        if config_path.exists():
            shutil.copy2(config_path, dest_dir / 'config.json')

    def _validate_template(self) -> None:
        """Validate that required template files exist."""
        if not self.template_path.exists():
            logger.error(f"Template file not found: {self.template_path}")
            logger.error(f"Expected location: {self.template_path.absolute()}")
            logger.error(f"OMRCHECKER_PATH: {OMRCHECKER_PATH}")
            logger.error(f"PROJECT_ROOT: {PROJECT_ROOT}")

            # Try alternative paths
            alt_paths = [
                PROJECT_ROOT / 'templates' / 'vidya_bharti_2025' / 'template.json',
                OMRCHECKER_PATH / 'samples' / 'sample1' / 'template.json',
            ]

            for alt in alt_paths:
                if alt.exists():
                    logger.warning(f"Found alternative template at: {alt}")
                    self.template_path = alt
                    return

            raise FileNotFoundError(
                f"Template file not found at {self.template_path}. "
                f"Please ensure the template directory is properly cloned."
            )
        else:
            logger.info(f"Template validated: {self.template_path}")

    def initialize_batch(self) -> bool:
        """
        Initialize batch processing directory.
        Call once before processing multiple images.
        Sets up template, reference image, and config once for the entire batch.
        """
        if OMRCheckerEngine._batch_initialized:
            return True

        try:
            # Create batch processing directory
            OMRCheckerEngine._batch_dir = WRITABLE_PATH / 'inputs' / '_batch_processing'
            OMRCheckerEngine._batch_dir.mkdir(parents=True, exist_ok=True)

            # Copy all template files (template.json, omr-filled.png, config.json)
            if self.template_path.exists():
                self._copy_template_files(OMRCheckerEngine._batch_dir)
                OMRCheckerEngine._batch_initialized = True
                logger.info("Batch processing initialized (template and reference image copied)")
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

        Uses FeatureBasedAlignment preprocessor for automatic image alignment:
        - Images of any dimension are automatically warped to match reference
        - No manual coordinate scaling needed
        - Reuses loaded OMRChecker modules for performance

        Args:
            image_path: Path to the OMR image

        Returns:
            ProcessingResult with extracted candidate data
        """
        start_time = time.time()

        try:
            # Clear previous results to prevent accumulation
            self._clear_results_directory()

            # Create a temporary directory for this image
            self._temp_dir = WRITABLE_PATH / 'inputs' / '_processing'
            self._temp_dir.mkdir(parents=True, exist_ok=True)

            # Copy template files (FeatureBasedAlignment handles dimension normalization)
            self._copy_template_files(self._temp_dir)
            logger.info(f"Processing image: {Path(image_path).name}")

            # Copy the image to temp directory
            image_name = Path(image_path).name
            temp_image = self._temp_dir / image_name
            shutil.copy2(image_path, temp_image)

            # Run OMRChecker (using pre-loaded modules)
            # Pass the image filename so we can match the correct row in CSV
            result_data = self._run_omrchecker_optimized(image_name)

            # Clean up temp files
            self._cleanup_temp_files()

            if result_data:
                logger.info(f"Successfully got result data for {image_path}")
                candidate = self._convert_to_candidate(result_data, image_path)
                processing_time = (time.time() - start_time) * 1000
                logger.info(f"Converted to candidate: name='{candidate.name}', roll='{candidate.roll_number}'")

                return ProcessingResult(
                    success=True,
                    candidate=candidate,
                    source_file=image_path,
                    processing_time_ms=processing_time
                )
            else:
                logger.warning(f"OMRChecker returned no results for {image_path}")
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

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary processing files."""
        try:
            if self._temp_dir and self._temp_dir.exists():
                for f in self._temp_dir.iterdir():
                    if f.is_file():
                        f.unlink()
        except Exception as e:
            logger.warning(f"Temp file cleanup error: {e}")

    def _clear_results_directory(self) -> None:
        """Clear the Results directory before processing each image.

        This prevents result accumulation from previous runs which causes
        the parser to read stale/wrong results.
        """
        try:
            results_dir = self._outputs_dir / 'Results'
            if results_dir.exists():
                for f in results_dir.iterdir():
                    if f.is_file() and f.suffix == '.csv':
                        f.unlink()
                        logger.debug(f"Cleared old results file: {f.name}")
        except Exception as e:
            logger.warning(f"Results cleanup error: {e}")

    def _run_omrchecker_optimized(self, image_filename: str = None) -> Optional[Dict[str, Any]]:
        """
        Run OMRChecker using pre-loaded modules (OPTIMIZED).
        Skips module import/unload cycle - saves ~3-5 seconds per image.

        Args:
            image_filename: The filename of the image being processed (for result matching)
        """
        global _omrchecker_entry_point

        try:
            if not _omrchecker_modules_loaded or _omrchecker_entry_point is None:
                logger.error("OMRChecker modules not loaded!")
                return self._run_omrchecker_fallback(image_filename)

            # Prepare arguments
            args = {
                'debug': False,
                'output_dir': str(self._outputs_dir),
                'autoAlign': False,
                'setLayout': False
            }

            # Run OMRChecker using pre-loaded entry point
            _omrchecker_entry_point(self._temp_dir, args)

            # Parse the results from CSV, matching by filename
            return self._parse_results(image_filename)

        except Exception as e:
            logger.error(f"Error running OMRChecker (optimized): {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fall back to traditional method
            return self._run_omrchecker_fallback(image_filename)

    def _run_omrchecker_fallback(self, image_filename: str = None) -> Optional[Dict[str, Any]]:
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

            return self._parse_results(image_filename)

        except Exception as e:
            logger.error(f"Error running OMRChecker (fallback): {e}")
            return None

    def _parse_results(self, image_filename: str = None) -> Optional[Dict[str, Any]]:
        """Parse OMRChecker CSV results.

        Args:
            image_filename: If provided, find the row matching this filename.
                          Otherwise returns the last row (legacy behavior).
        """
        try:
            results_dir = self._outputs_dir / 'Results'
            logger.info(f"Looking for results in: {results_dir}")
            logger.info(f"Results directory exists: {results_dir.exists()}")

            if results_dir.exists():
                all_files = list(results_dir.iterdir())
                logger.info(f"Files in results directory: {[f.name for f in all_files]}")

            csv_files = list(results_dir.glob('Results_*.csv'))
            logger.info(f"Found {len(csv_files)} CSV files matching 'Results_*.csv'")

            if csv_files:
                csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using most recent CSV file: {csv_file}")
                logger.info(f"Looking for image: {image_filename}")

                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    result_row = None
                    matched_row = None
                    row_count = 0

                    for row in reader:
                        result_row = dict(row)
                        row_count += 1

                        # Match by filename if provided (column is 'file_id' in CSV)
                        if image_filename:
                            row_filename = row.get('file_id', '')
                            if row_filename == image_filename:
                                matched_row = dict(row)
                                logger.info(f"Found matching row for {image_filename}")

                    logger.info(f"Parsed {row_count} rows from CSV")

                    # Return matched row if found, otherwise fallback to last row
                    if matched_row:
                        logger.info(f"Returning matched row for: {image_filename}")
                        return matched_row
                    elif result_row:
                        logger.warning(f"No matching row found for {image_filename}, "
                                      f"returning last row (file_id={result_row.get('file_id', 'unknown')})")
                        return result_row

            logger.warning(f"No CSV results found in {results_dir}")
            # List what's in the outputs directory
            if self._outputs_dir.exists():
                output_contents = list(self._outputs_dir.iterdir())
                logger.warning(f"Contents of outputs dir: {[f.name for f in output_contents]}")
            return None

        except Exception as e:
            logger.error(f"Error parsing results: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


    def _clean_detected_name(self, raw_name: str) -> str:
        """Clean up detected name by removing spurious characters.

        Common issues:
        - Trailing 'W' characters from false positives in the W row
        - Multiple consecutive same letters (unlikely in real names)
        - Single isolated characters at the end
        - Noise characters appearing at specific column positions

        Args:
            raw_name: The raw detected name

        Returns:
            Cleaned name string
        """
        import re

        if not raw_name:
            return raw_name

        name = raw_name

        # Step 1: Remove trailing W's (common false positive)
        # Pattern: one or more W's at the end, possibly with spaces before
        name = re.sub(r'\s*W+\s*$', '', name)

        # Step 2: Remove W attached to end of words (like "KUMARW")
        # This catches cases where W is falsely detected right after the last letter
        name = re.sub(r'([A-VX-Z])W(\s|$)', r'\1\2', name)

        # Step 3: Remove multiple consecutive same letters (2+ in a row) for W and M
        # These are commonly falsely detected. Keep doubles only for other letters.
        name = re.sub(r'W{2,}', '', name)  # Remove WW, WWW, WWWW etc
        name = re.sub(r'M{2,}', 'M', name)  # Keep only one M

        # Step 4: Remove other multiple consecutive same letters (3+ in a row)
        # This catches patterns like "LLL" which are unlikely in names
        def remove_triple_plus(match):
            char = match.group(1)
            return char * 2
        name = re.sub(r'([A-LN-VX-Z])\1{2,}', remove_triple_plus, name)

        # Step 5: Remove noise patterns like "WM", "MW" which are unlikely combinations
        # after spaces (indicates detection of non-filled bubbles)
        name = re.sub(r'\s+[WM]{2,}', '', name)

        # Step 6: Remove isolated single characters at the end after multiple spaces
        name = re.sub(r'\s{2,}[A-Z]$', '', name)

        # Step 7: Clean up excessive internal spaces
        name = re.sub(r'\s{3,}', '  ', name)  # Max 2 spaces between words

        # Step 8: Clean trailing single letters after space at very end
        # (like "KUMAR K" where K is noise)
        name = re.sub(r'\s+[JKWM]$', '', name)

        # Step 9: Final strip and cleanup
        name = name.strip()

        # Remove any remaining trailing single space + letter combinations
        while re.search(r'\s{2,}[A-Z]$', name):
            name = re.sub(r'\s{2,}[A-Z]$', '', name).strip()

        return name

    def _convert_to_candidate(self, result_data: Dict[str, Any], source_image: str) -> Candidate:
        """Convert OMRChecker CSV row to Candidate model."""
        candidate = Candidate(source_image=source_image)

        # Extract Candidate Name (name1..name25)
        has_individual_cols = any(f'name{i}' in result_data for i in range(1, 26))

        if has_individual_cols:
            # Build name from individual columns, treating empty columns as spaces
            name_chars = []

            for i in range(1, 26):
                key = f'name{i}'
                if key in result_data:
                    char = str(result_data[key]).strip() if result_data[key] else ''
                    if char:
                        # Valid character detected
                        # Handle multi-mark: take first letter only
                        if len(char) > 1:
                            char = char[0]
                        name_chars.append(char)
                    else:
                        # Empty/blank column - treat as space
                        name_chars.append(' ')
                else:
                    # Column not in data - treat as space
                    name_chars.append(' ')

            # Join all characters and strip trailing spaces only
            # This preserves internal spaces while removing trailing blanks
            full_name = ''.join(name_chars)
            raw_name = full_name.rstrip()

            # Clean up spurious detections
            candidate.name = self._clean_detected_name(raw_name)

            logger.debug(f"Name extraction: raw chars = {name_chars}")
            logger.debug(f"Name extraction: raw name = '{raw_name}'")
            logger.debug(f"Name extraction: cleaned name = '{candidate.name}'")
        elif 'CandidateName' in result_data:
            raw_name = str(result_data['CandidateName']).strip()
            candidate.name = self._clean_detected_name(raw_name)
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

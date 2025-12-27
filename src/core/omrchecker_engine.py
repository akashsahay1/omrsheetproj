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
        """Clean up detected name by removing spurious characters and adding spaces.

        Args:
            raw_name: The raw detected name

        Returns:
            Cleaned name string
        """
        import re

        if not raw_name:
            return raw_name

        name = raw_name.strip().upper()

        # Remove all spaces first to normalize, then re-add correctly
        name_no_space = name.replace(' ', '')

        # Comprehensive list of Indian surnames - KUMARI before KUMAR for proper matching
        # Pre-sorted by length descending for longest-match-first semantics
        surnames_by_len = [
            'SRIVASTAVA', 'CHATTERJEE', 'MUKHERJEE', 'BANERJEE',  # 10
            'KERKETTA', 'HEMBRAM', 'RAJWAR', 'MAHATO',  # 7-8
            'BHARTI', 'MANDAL', 'CHANDRA', 'KUMARI', 'PRASAD', 'SHARMA', 'MISHRA',  # 6
            'PANDEY', 'TIWARI', 'HANSDA', 'TIRKEY', 'KHATUN', 'PATHAK', 'ANSARI',
            'SHUKLA', 'DUBEY', 'VERMA', 'PANDIT', 'TOPPO', 'LAKRA', 'BASKI', 'SAHAY',
            'KISKU', 'SOREN', 'ORAON', 'GUPTA', 'YADAV', 'KACHHAP', 'MARANDI',  # 5-7
            'KUMAR', 'SINGH', 'MURMU', 'SHAW', 'TUDU', 'MODI', 'TURI', 'DEVI', 'RANI',
            'MUNDA', 'HORO', 'EKKA', 'MINJ', 'BECK', 'TIGGA', 'KUJUR', 'BARA',  # Tribal
            'DAS', 'PAL', 'LAL', 'RAM', 'SAH', 'ROY',  # 3
        ]
        # Also keep the list sorted for safety
        surnames_sorted = sorted(surnames_by_len, key=len, reverse=True)

        # Step 1: Remove trailing garbage patterns
        # First, collapse repeated M/W (MMMM -> M, WWWW -> W)
        name_no_space = re.sub(r'M{2,}', 'M', name_no_space)
        name_no_space = re.sub(r'W{2,}', 'W', name_no_space)
        # Then remove trailing M, W patterns (including mixed like MWMW)
        name_no_space = re.sub(r'[MW]+$', '', name_no_space)
        # Remove patterns like MK, WK, KM at the end (common garbage)
        name_no_space = re.sub(r'[MW]K$', '', name_no_space)
        name_no_space = re.sub(r'K[MW]$', '', name_no_space)

        # Step 2: Remove garbage AFTER known surnames at end
        # Pattern: surname + any short garbage (2-6 chars) at end
        for surname in surnames_sorted:
            # Look for surname followed by short garbage at end
            pattern = rf'({surname})([A-Z]{{2,6}})$'  # At least 2 chars of garbage
            match = re.search(pattern, name_no_space)
            if match:
                garbage = match.group(2)
                # Check if this "garbage" could extend a shorter surname to a longer one
                # e.g., KUMAR + I = KUMARI (valid surname, don't strip)
                full_word = surname + garbage
                is_valid_extension = any(full_word.endswith(s) and s != surname
                                        for s in surnames_sorted if len(s) > len(surname))
                if is_valid_extension:
                    continue  # Don't strip, it's a valid longer surname

                # Only strip if garbage looks like noise (has MW or repeated chars)
                if (any(c in garbage for c in 'MW') or
                    len(garbage) <= 3 or
                    re.search(r'(.)\1', garbage)):  # repeated char
                    name_no_space = name_no_space[:match.start()] + surname
                    break

        # Step 3: Find and extract surnames from the end (handles compound surnames)
        found_surnames = []
        remaining = name_no_space

        for iteration in range(3):  # Max 3 surnames
            found_one = False

            # First, check if any surname is at the end
            for surname in surnames_sorted:
                if remaining.endswith(surname):
                    # For short surnames (<=4 chars), ensure what remains is substantial
                    # This prevents false matches like RAM in VIKRAM
                    potential_remaining = remaining[:-len(surname)]
                    potential_remaining_clean = re.sub(r'[MW]+$', '', potential_remaining)

                    # Skip if short surname would leave very short remainder (likely false match)
                    if len(surname) <= 4 and len(potential_remaining_clean) <= 3:
                        continue

                    found_surnames.insert(0, surname)
                    remaining = potential_remaining
                    # Remove trailing garbage: multiple M/W (2+)
                    remaining = re.sub(r'[MW]{2,}$', '', remaining)
                    # Also remove single W (rare in names)
                    if remaining.endswith('W'):
                        remaining = remaining[:-1]
                    # Remove single M only if NOT a common name ending
                    # Many Indian names end in "AM" (VIKRAM, SHYAM, RAM) or "UM" (GAYTUM)
                    # Only strip M if it doesn't form these patterns
                    if remaining.endswith('M') and len(remaining) > 4:
                        # Keep M if it forms common name endings: -AM, -UM, -IM
                        if not (remaining.endswith('AM') or remaining.endswith('UM') or remaining.endswith('IM')):
                            remaining = remaining[:-1]
                    found_one = True
                    break

            if not found_one and iteration > 0:
                # No surname at end - check for EMBEDDED surname with short garbage after
                # This handles cases like KUMARAMANDAL where A is garbage between KUMAR and MANDAL
                for surname in surnames_sorted:
                    idx = remaining.rfind(surname)
                    if idx > 0:  # Found embedded (not at start)
                        after = remaining[idx + len(surname):]
                        # If what's after is short garbage (1-5 chars), extract this surname
                        if 0 < len(after) <= 5 and re.match(r'^[A-Z]*$', after):
                            found_surnames.insert(0, surname)
                            remaining = remaining[:idx]
                            # Don't strip M/W here - could be part of name (e.g., RADHESHYAM)
                            found_one = True
                            break

            if not found_one:
                break

        # Step 4: Clean the first name part
        first_name = remaining

        # Remove trailing garbage: only multiple M/W (2+) or single W
        # Single M is often part of names (VIKRAM, RAM, SHYAM)
        first_name = re.sub(r'[MW]{2,}$', '', first_name)
        if first_name.endswith('W'):
            first_name = first_name[:-1]

        # Handle double A at end (detection error: DEEPIKAA -> DEEPIKA)
        # But keep single A (SNEHA stays SNEHA)
        if len(first_name) > 4 and first_name.endswith('AA'):
            first_name = first_name[:-1]

        # Step 5: Build the final name
        parts = []
        if first_name and len(first_name) >= 2:
            parts.append(first_name)
        parts.extend(found_surnames)

        name = ' '.join(parts)

        # Step 6: Clean up repeated characters
        name = re.sub(r'M{2,}', 'M', name)
        name = re.sub(r'W{2,}', 'W', name)
        name = re.sub(r'([A-Z])\1{3,}', r'\1', name)  # 4+ same letters -> 1

        # Step 7: Remove garbage patterns (XX)
        name = re.sub(r'X{2,}[A-Z]*', '', name)

        # Normalize spaces
        name = ' '.join(name.split())

        return name

    def _convert_to_candidate(self, result_data: Dict[str, Any], source_image: str) -> Candidate:
        """Convert OMRChecker CSV row to Candidate model."""
        candidate = Candidate(source_image=source_image)

        # Extract Candidate Name
        # Check various possible column names for the name field
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
        elif 'Name' in result_data and result_data['Name']:
            # Simple 'Name' column from OMRChecker customLabels
            raw_name = str(result_data['Name']).strip()
            candidate.name = self._clean_detected_name(raw_name)
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

        # Extract Class (check both cases)
        class_val = result_data.get('Class') or result_data.get('class')
        if class_val:
            candidate.class_ = str(class_val).strip()

        # Extract Section (check both cases)
        section_val = result_data.get('Section') or result_data.get('section')
        if section_val:
            candidate.section = str(section_val).strip()

        # Extract School Code (check multiple possible column names)
        school_val = result_data.get('School') or result_data.get('SchoolCode') or result_data.get('school_code')
        if school_val:
            candidate.school_code = str(school_val).strip()
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

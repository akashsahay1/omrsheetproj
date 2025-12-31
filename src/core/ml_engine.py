"""
ML-based OMR Processing Engine

Uses YOLO processor with ML bubble classification and OCR fallback
for high-accuracy OMR processing (98%+ name detection accuracy).
"""
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

from src.models.candidate import Candidate
from src.models.result import ProcessingResult
from src.utils.logging_utils import get_logger

logger = get_logger('ml_engine')


def get_base_path() -> Path:
    """Get base path - works for both dev and bundled app."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    else:
        return Path(__file__).parent.parent.parent


def get_models_path() -> Path:
    """Get path to ML models directory."""
    base = get_base_path()
    return base / 'models'


class MLEngine:
    """
    ML-based OMR processing engine.

    Uses YOLOOMRProcessor with:
    - ML bubble classifier for improved detection
    - OCR fallback for low-confidence names
    - 98%+ accuracy on name detection

    LAZY LOADING: Models are not loaded until first use to improve startup time.
    """

    def __init__(self, lazy: bool = True):
        """Initialize the ML engine.

        Args:
            lazy: If True, defer model loading until first use (faster startup)
        """
        self._processor = None
        self._initialized = False
        self._initializing = False
        self._init_error = None

        # Paths
        self._base_path = get_base_path()
        self._models_path = get_models_path()

        # Reference image for alignment
        self._reference_path = self._base_path / 'OMRChecker' / 'inputs' / 'VidyaBharti2025' / 'omr-filled.png'

        # ML model paths - let YOLOOMRProcessor choose the best available model
        # (it will prioritize v4, v3, v2 in that order)
        self._bubble_classifier_path = None

        # Only initialize immediately if not lazy loading
        if not lazy:
            self._initialize()

    def _initialize(self):
        """Initialize the YOLO processor with ML components."""
        try:
            from src.ai.yolo_omr_processor import YOLOOMRProcessor

            # Check if reference image exists
            if not self._reference_path.exists():
                logger.warning(f"Reference image not found: {self._reference_path}")
                # Try alternate locations
                alt_paths = [
                    self._base_path / 'templates' / 'omr-filled.png',
                    self._base_path / 'omr-filled.png',
                ]
                for alt in alt_paths:
                    if alt.exists():
                        self._reference_path = alt
                        break

            # Check bubble classifier - pass None to let YOLOOMRProcessor auto-select
            # (it will prioritize v4 > v3 > v2)
            bubble_path = None
            if self._bubble_classifier_path is not None and self._bubble_classifier_path.exists():
                bubble_path = str(self._bubble_classifier_path)
                logger.info(f"Using ML bubble classifier: {bubble_path}")

            # Initialize processor (passing None for bubble_classifier_path enables auto-selection)
            self._processor = YOLOOMRProcessor(
                reference_image_path=str(self._reference_path) if self._reference_path.exists() else None,
                bubble_classifier_path=bubble_path
            )

            self._initialized = True
            logger.info("ML Engine initialized successfully")

        except Exception as e:
            self._init_error = str(e)
            logger.error(f"ML Engine initialization failed: {e}")
            import traceback
            traceback.print_exc()

    def is_initialized(self) -> bool:
        """Check if engine is ready."""
        return self._initialized

    def is_initializing(self) -> bool:
        """Check if engine is currently initializing."""
        return self._initializing

    def get_init_error(self) -> Optional[str]:
        """Get initialization error if any."""
        return self._init_error

    def ensure_initialized(self) -> bool:
        """Ensure engine is initialized (lazy loading).

        Call this before processing. Will initialize on first call.

        Returns:
            True if initialized successfully, False otherwise
        """
        if self._initialized:
            return True

        if self._init_error:
            return False

        if not self._initializing:
            self._initialize()

        return self._initialized

    def initialize_async(self, callback=None):
        """Initialize engine in background thread.

        Args:
            callback: Optional function to call when initialization completes.
                      Will be called with (success: bool, error: Optional[str])
        """
        if self._initialized or self._initializing:
            if callback:
                callback(self._initialized, self._init_error)
            return None

        import threading
        self._initializing = True

        def init_thread():
            try:
                self._initialize()
            finally:
                self._initializing = False
            if callback:
                callback(self._initialized, self._init_error)

        thread = threading.Thread(target=init_thread, daemon=True)
        thread.start()
        return thread

    def process_image(self, image_path: str) -> ProcessingResult:
        """
        Process a single OMR image.

        Args:
            image_path: Path to the OMR image

        Returns:
            ProcessingResult with candidate data
        """
        start_time = time.time()

        # Auto-initialize on first use (lazy loading)
        if not self._initialized:
            if not self.ensure_initialized():
                return ProcessingResult(
                    success=False,
                    error_message=f"ML Engine not initialized: {self._init_error}"
                )

        try:
            # Process with YOLO processor
            result = self._processor.process(image_path)

            if not result.success:
                return ProcessingResult(
                    success=False,
                    error_message=result.error_message or "Processing failed"
                )

            # Find multi-marked questions (answers with multiple letters)
            multi_marked_questions = [
                q for q, ans in result.answers.items()
                if ans and len(ans) > 1
            ]

            # Get detection confidence from result
            detection_conf = getattr(result, 'confidence', 100.0)

            # Convert to Candidate object
            candidate = Candidate(
                name=result.name or "",
                roll_number=result.roll_number or "",
                class_=result.class_ or "",
                section=result.section or "",
                school_code=result.school_code or "",
                answers=result.answers or {},
                multi_marked_questions=multi_marked_questions,
                source_image=image_path,
                detection_confidence=detection_conf
            )

            processing_time = int((time.time() - start_time) * 1000)

            # Get detection confidence from OMR result (0-100 scale)
            detection_conf = getattr(result, 'confidence', 100)

            return ProcessingResult(
                success=True,
                candidate=candidate,
                processing_time_ms=processing_time,
                warnings=["OCR fallback used"] if getattr(result, 'ocr_used', False) else [],
                detection_confidence=detection_conf,
                needs_manual_review=detection_conf < 70  # Flag low confidence for review
            )

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            import traceback
            traceback.print_exc()

            return ProcessingResult(
                success=False,
                error_message=str(e)
            )

    def initialize_batch(self) -> None:
        """Initialize batch processing (no-op for ML engine)."""
        logger.info("ML Engine batch processing initialized")

    def finalize_batch(self) -> None:
        """Finalize batch processing (no-op for ML engine)."""
        logger.info("ML Engine batch processing finalized")

    def process_batch(self, image_paths: list, callback=None) -> list:
        """
        Process multiple images.

        Args:
            image_paths: List of image paths
            callback: Optional callback(index, total, result) for progress

        Returns:
            List of ProcessingResult objects
        """
        results = []
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            result = self.process_image(path)
            results.append(result)

            if callback:
                callback(i + 1, total, result)

        return results

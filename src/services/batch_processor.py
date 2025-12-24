"""
Batch processing service with progress tracking
"""
import threading
import time
from pathlib import Path
from typing import List, Optional, Callable, Dict, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from src.core.omrchecker_engine import OMRCheckerEngine as OMREngine
from src.models.result import ProcessingResult, BatchResult, EvaluationResult
from src.models.candidate import Candidate
from src.models.answer_key import AnswerKey
from src.services.scorer import Scorer
from src.services.excel_exporter import ExcelExporter
from src.services.validator import Validator
from src.utils.file_utils import FileUtils
from src.utils.logging_utils import get_logger

logger = get_logger('batch_processor')


class BatchStatus(Enum):
    """Batch processing status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class ProcessingProgress:
    """Progress tracking data"""
    status: BatchStatus = BatchStatus.IDLE
    current: int = 0
    total: int = 0
    percentage: float = 0.0
    current_file: str = ""
    message: str = ""
    start_time: Optional[float] = None
    elapsed_seconds: float = 0.0
    estimated_remaining: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'status': self.status.value,
            'current': self.current,
            'total': self.total,
            'percentage': round(self.percentage, 1),
            'current_file': self.current_file,
            'message': self.message,
            'elapsed_seconds': round(self.elapsed_seconds, 1),
            'estimated_remaining': round(self.estimated_remaining, 1),
            'errors': self.errors
        }


class BatchProcessor:
    """
    Processes multiple OMR sheets with progress tracking and cancellation support

    Features:
    - Background thread processing
    - Real-time progress updates
    - Cancellation support
    - Error handling with skip capability
    - Optional scoring and export
    """

    def __init__(self, engine: Optional[OMREngine] = None):
        """
        Initialize batch processor

        Args:
            engine: OMR engine instance (creates default if not provided)
        """
        self.engine = engine or OMREngine()
        self.scorer = Scorer()
        self.exporter = ExcelExporter()

        # Processing state
        self._progress = ProcessingProgress()
        self._is_cancelled = False
        self._is_paused = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Results storage
        self._batch_result: Optional[BatchResult] = None
        self._progress_callback: Optional[Callable[[ProcessingProgress], None]] = None

        # Exam settings for export
        self._exam_level: str = 'Anshik'
        self._exam_type: str = 'Chayan'

    def set_answer_key(self, answer_key: AnswerKey) -> None:
        """Set answer key for scoring"""
        self.scorer.set_answer_key(answer_key)

    def load_answer_key(self, file_path: str) -> bool:
        """Load answer key from file"""
        return self.scorer.load_answer_key(file_path)

    def start(self,
              input_folder: str,
              output_path: Optional[str] = None,
              progress_callback: Optional[Callable[[ProcessingProgress], None]] = None,
              with_scoring: bool = True,
              exam_level: str = 'Anshik',
              exam_type: str = 'Chayan') -> bool:
        """
        Start batch processing in background thread

        Args:
            input_folder: Folder containing OMR images
            output_path: Optional Excel output path
            progress_callback: Optional callback for progress updates
            with_scoring: Whether to score results (requires answer key)
            exam_level: Exam level (Anshik/Mukhya)
            exam_type: Exam type (Chayan/Khoj)

        Returns:
            True if processing started successfully
        """
        # Store exam settings
        self._exam_level = exam_level
        self._exam_type = exam_type
        # Check if already running
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Batch processing already in progress")
            return False

        # Get image files
        image_files = FileUtils.get_image_files(input_folder)

        if not image_files:
            logger.error(f"No image files found in {input_folder}")
            self._update_progress(status=BatchStatus.ERROR,
                                 message="No image files found")
            return False

        # Validate
        validation = Validator.validate_batch_input(input_folder, image_files)
        if not validation.is_valid:
            logger.error(f"Validation failed: {validation.errors}")
            self._update_progress(status=BatchStatus.ERROR,
                                 message="; ".join(validation.errors))
            return False

        # Reset state
        self._is_cancelled = False
        self._is_paused = False
        self._progress_callback = progress_callback
        self._batch_result = BatchResult(total_sheets=len(image_files))

        # Initialize progress
        self._update_progress(
            status=BatchStatus.RUNNING,
            total=len(image_files),
            current=0,
            message="Starting batch processing..."
        )

        # Start processing thread
        self._thread = threading.Thread(
            target=self._process_batch,
            args=(image_files, output_path, with_scoring),
            daemon=True
        )
        self._thread.start()

        logger.info(f"Started batch processing: {len(image_files)} files")
        return True

    def _process_batch(self,
                        image_files: List[str],
                        output_path: Optional[str],
                        with_scoring: bool) -> None:
        """
        Internal batch processing method (runs in background thread)
        """
        start_time = time.time()
        self._progress.start_time = start_time
        self._batch_result.start_time = datetime.now().isoformat()

        processed_count = 0
        candidates = []

        # Initialize batch processing for optimization
        # This copies template once instead of per-image
        self.engine.initialize_batch()

        try:
            for idx, image_path in enumerate(image_files):
                # Check for cancellation
                if self._is_cancelled:
                    self._update_progress(
                        status=BatchStatus.CANCELLED,
                        message="Processing cancelled by user"
                    )
                    break

                # Check for pause
                while self._is_paused:
                    time.sleep(0.1)
                    if self._is_cancelled:
                        break

                # Update progress
                filename = Path(image_path).name
                self._update_progress(
                    current=idx,
                    current_file=filename,
                    message=f"Processing {filename}..."
                )

                # Process image
                try:
                    result = self.engine.process_image(image_path)
                    self._batch_result.add_result(result)

                    if result.success and result.candidate:
                        result.candidate.image_index = idx
                        candidates.append(result.candidate)
                        logger.info(f"Successfully processed {filename}: "
                                   f"name='{result.candidate.name}', "
                                   f"roll='{result.candidate.roll_number}'")
                    else:
                        logger.warning(f"Failed to process {filename}: {result.error_message}")

                    if not result.success:
                        self._progress.errors.append(
                            f"{filename}: {result.error_message}"
                        )

                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    self._progress.errors.append(f"{filename}: {str(e)}")

                processed_count += 1

                # Update elapsed time and estimate
                elapsed = time.time() - start_time
                if processed_count > 0:
                    avg_time = elapsed / processed_count
                    remaining = (len(image_files) - processed_count) * avg_time
                    self._progress.elapsed_seconds = elapsed
                    self._progress.estimated_remaining = remaining

            # Finalize if not cancelled
            if not self._is_cancelled:
                logger.info(f"Batch processing complete: {len(candidates)} candidates collected")

                # Score if requested
                if with_scoring and self.scorer.answer_key:
                    self._update_progress(message="Calculating scores...")
                    logger.info(f"Scoring {len(candidates)} candidates with answer key")
                    evaluation_results = self.scorer.evaluate_batch(candidates)
                    self._batch_result.evaluation_results = evaluation_results
                    logger.info(f"Scoring complete: {len(evaluation_results)} evaluation results")
                else:
                    logger.warning(f"Skipping scoring: with_scoring={with_scoring}, "
                                  f"answer_key_set={self.scorer.answer_key is not None}")

                # Export if path provided
                if output_path:
                    self._update_progress(message="Exporting to Excel...")

                    if with_scoring and self._batch_result.evaluation_results:
                        self.exporter.export(
                            self._batch_result.evaluation_results,
                            output_path,
                            exam_level=self._exam_level,
                            exam_type=self._exam_type
                        )
                    else:
                        self.exporter.export_simple(candidates, output_path)

                # Calculate statistics
                self._batch_result.calculate_statistics()
                self._batch_result.end_time = datetime.now().isoformat()
                self._batch_result.total_time_seconds = time.time() - start_time

                self._update_progress(
                    status=BatchStatus.COMPLETED,
                    current=len(image_files),
                    percentage=100.0,
                    message=f"Completed: {self._batch_result.success_count} successful, "
                           f"{self._batch_result.failed_count} failed"
                )

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            self._update_progress(
                status=BatchStatus.ERROR,
                message=f"Error: {str(e)}"
            )

        finally:
            # Clean up batch processing directory
            self.engine.finalize_batch()

    def _update_progress(self, **kwargs) -> None:
        """Update progress and notify callback"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._progress, key):
                    setattr(self._progress, key, value)

            # Calculate percentage
            if self._progress.total > 0:
                self._progress.percentage = (
                    (self._progress.current / self._progress.total) * 100
                )

        # Notify callback
        if self._progress_callback:
            try:
                self._progress_callback(self._progress)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def cancel(self) -> None:
        """Cancel ongoing processing"""
        self._is_cancelled = True
        logger.info("Batch processing cancellation requested")

    def pause(self) -> None:
        """Pause processing"""
        self._is_paused = True
        self._update_progress(status=BatchStatus.PAUSED, message="Paused")

    def resume(self) -> None:
        """Resume paused processing"""
        self._is_paused = False
        self._update_progress(status=BatchStatus.RUNNING, message="Resumed")

    def get_progress(self) -> ProcessingProgress:
        """Get current progress"""
        with self._lock:
            return self._progress

    def get_results(self) -> Optional[BatchResult]:
        """Get batch results (available after completion)"""
        return self._batch_result

    def is_running(self) -> bool:
        """Check if processing is currently running"""
        return (self._thread is not None and
                self._thread.is_alive() and
                not self._is_cancelled)

    def is_completed(self) -> bool:
        """Check if processing completed"""
        return self._progress.status == BatchStatus.COMPLETED

    def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for processing to complete

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if completed, False if timed out or cancelled
        """
        if self._thread is None:
            return True

        self._thread.join(timeout=timeout)
        return self.is_completed()

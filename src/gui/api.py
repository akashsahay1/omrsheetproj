"""
PyWebView API - Python-JavaScript bridge
"""
import os
import json
import webview
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.core.omr_engine import OMREngine
from src.services.batch_processor import BatchProcessor, ProcessingProgress
from src.services.scorer import Scorer
from src.services.excel_exporter import ExcelExporter
from src.models.answer_key import AnswerKey
from src.models.result import EvaluationResult
from src.config.settings import Settings
from src.config.constants import APP_NAME, APP_VERSION
from src.utils.file_utils import FileUtils
from src.utils.logging_utils import get_logger

logger = get_logger('api')


class OMRApi:
    """
    API exposed to JavaScript via pywebview.api

    Handles all backend operations called from the web UI
    """

    def __init__(self, window: Optional[webview.Window] = None):
        """Initialize API with optional window reference"""
        self.window = window
        self.settings = Settings.load()

        # Initialize components
        self.engine = OMREngine()
        self.batch_processor = BatchProcessor(self.engine)
        self.scorer = Scorer()
        self.exporter = ExcelExporter()

        # State
        self._current_results: List[EvaluationResult] = []
        self._answer_key: Optional[AnswerKey] = None

    def set_window(self, window: webview.Window) -> None:
        """Set window reference"""
        self.window = window

    # === Application Info ===

    def get_app_info(self) -> Dict[str, Any]:
        """Get application information"""
        return {
            'name': APP_NAME,
            'version': APP_VERSION,
            'status': 'ready'
        }

    # === File Dialogs ===

    def select_input_folder(self) -> Dict[str, Any]:
        """Open folder picker for input images"""
        try:
            result = self.window.create_file_dialog(
                webview.FOLDER_DIALOG,
                directory=self.settings.last_input_folder or str(Path.home())
            )

            if result and len(result) > 0:
                folder = result[0]
                self.settings.last_input_folder = folder
                self.settings.save()

                # Get file count
                files = FileUtils.get_image_files(folder)

                return {
                    'success': True,
                    'path': folder,
                    'file_count': len(files)
                }

            return {'success': False, 'message': 'No folder selected'}

        except Exception as e:
            logger.error(f"Folder selection error: {e}")
            return {'success': False, 'message': str(e)}

    def select_output_file(self) -> Dict[str, Any]:
        """Open file picker for Excel output"""
        try:
            result = self.window.create_file_dialog(
                webview.SAVE_DIALOG,
                directory=self.settings.last_output_folder or str(Path.home()),
                save_filename='omr_results.xlsx',
                file_types=('Excel Files (*.xlsx)',)
            )

            if result:
                path = result if isinstance(result, str) else result[0]

                # Ensure .xlsx extension
                if not path.endswith('.xlsx'):
                    path += '.xlsx'

                self.settings.last_output_folder = str(Path(path).parent)
                self.settings.save()

                return {'success': True, 'path': path}

            return {'success': False, 'message': 'No file selected'}

        except Exception as e:
            logger.error(f"File selection error: {e}")
            return {'success': False, 'message': str(e)}

    def select_answer_key_file(self) -> Dict[str, Any]:
        """Open file picker for answer key"""
        try:
            result = self.window.create_file_dialog(
                webview.OPEN_DIALOG,
                directory=self.settings.last_answer_key_path or str(Path.home()),
                file_types=(
                    'All Supported (*.xlsx;*.csv;*.json)',
                    'Excel Files (*.xlsx)',
                    'CSV Files (*.csv)',
                    'JSON Files (*.json)'
                )
            )

            if result and len(result) > 0:
                path = result[0]
                return self.load_answer_key(path)

            return {'success': False, 'message': 'No file selected'}

        except Exception as e:
            logger.error(f"Answer key selection error: {e}")
            return {'success': False, 'message': str(e)}

    # === Answer Key Operations ===

    def load_answer_key(self, file_path: str) -> Dict[str, Any]:
        """Load answer key from file"""
        try:
            self._answer_key = AnswerKey.from_file(file_path)
            self.scorer.set_answer_key(self._answer_key)
            self.batch_processor.set_answer_key(self._answer_key)

            self.settings.last_answer_key_path = file_path
            self.settings.save()

            return {
                'success': True,
                'path': file_path,
                'filled_count': self._answer_key.get_filled_count(),
                'is_complete': self._answer_key.is_complete(),
                'answers': self._answer_key.answers
            }

        except Exception as e:
            logger.error(f"Answer key load error: {e}")
            return {'success': False, 'message': str(e)}

    def set_manual_answer_key(self, answers: Dict[str, str]) -> Dict[str, Any]:
        """Set answer key from manual input"""
        try:
            self._answer_key = AnswerKey()

            for q_str, answer in answers.items():
                q_num = int(q_str)
                self._answer_key.set_answer(q_num, answer.upper())

            self.scorer.set_answer_key(self._answer_key)
            self.batch_processor.set_answer_key(self._answer_key)

            return {
                'success': True,
                'filled_count': self._answer_key.get_filled_count(),
                'is_complete': self._answer_key.is_complete()
            }

        except Exception as e:
            logger.error(f"Manual answer key error: {e}")
            return {'success': False, 'message': str(e)}

    def get_answer_key(self) -> Dict[str, Any]:
        """Get current answer key"""
        if self._answer_key is None:
            return {'success': False, 'message': 'No answer key loaded'}

        return {
            'success': True,
            'answers': self._answer_key.answers,
            'filled_count': self._answer_key.get_filled_count(),
            'is_complete': self._answer_key.is_complete()
        }

    # === Processing Operations ===

    def start_processing(self,
                         input_folder: str,
                         output_path: Optional[str] = None,
                         with_scoring: bool = True) -> Dict[str, Any]:
        """Start batch processing"""
        try:
            # Validate input
            if not input_folder or not os.path.isdir(input_folder):
                return {'success': False, 'message': 'Invalid input folder'}

            if with_scoring and self._answer_key is None:
                return {'success': False, 'message': 'Answer key required for scoring'}

            # Start processing
            started = self.batch_processor.start(
                input_folder=input_folder,
                output_path=output_path,
                with_scoring=with_scoring
            )

            return {
                'success': started,
                'message': 'Processing started' if started else 'Failed to start'
            }

        except Exception as e:
            logger.error(f"Start processing error: {e}")
            return {'success': False, 'message': str(e)}

    def cancel_processing(self) -> Dict[str, Any]:
        """Cancel ongoing processing"""
        try:
            self.batch_processor.cancel()
            return {'success': True, 'message': 'Cancellation requested'}
        except Exception as e:
            return {'success': False, 'message': str(e)}

    def get_progress(self) -> Dict[str, Any]:
        """Get current processing progress"""
        progress = self.batch_processor.get_progress()
        return progress.to_dict()

    def is_processing(self) -> bool:
        """Check if processing is running"""
        return self.batch_processor.is_running()

    # === Results Operations ===

    def get_results(self) -> Dict[str, Any]:
        """Get all processing results"""
        batch_result = self.batch_processor.get_results()

        if batch_result is None:
            return {'success': False, 'message': 'No results available'}

        # Store results for later access
        self._current_results = batch_result.evaluation_results

        results_data = []
        for result in batch_result.evaluation_results:
            if result.candidate:
                results_data.append({
                    'name': result.candidate.name,
                    'roll_number': result.candidate.roll_number,
                    'class': result.candidate.class_,
                    'section': result.candidate.section,
                    'school_code': result.candidate.school_code,
                    'correct': result.correct_count,
                    'wrong': result.wrong_count,
                    'unanswered': result.unanswered_count,
                    'multi_marked': result.multi_marked_count,
                    'total_score': result.total_score,
                    'percentage': result.percentage,
                    'rank': result.rank
                })

        return {
            'success': True,
            'results': results_data,
            'total_count': len(results_data),
            'statistics': {
                'success_count': batch_result.success_count,
                'failed_count': batch_result.failed_count,
                'average_score': batch_result.average_score,
                'highest_score': batch_result.highest_score,
                'lowest_score': batch_result.lowest_score
            }
        }

    def get_result_detail(self, index: int) -> Dict[str, Any]:
        """Get detailed result for a specific candidate"""
        if index < 0 or index >= len(self._current_results):
            return {'success': False, 'message': 'Invalid index'}

        result = self._current_results[index]

        if result.candidate is None:
            return {'success': False, 'message': 'No candidate data'}

        return {
            'success': True,
            'candidate': result.candidate.to_dict(),
            'evaluation': result.to_dict()
        }

    def export_results(self, output_path: str) -> Dict[str, Any]:
        """Export results to Excel"""
        try:
            if not self._current_results:
                return {'success': False, 'message': 'No results to export'}

            success = self.exporter.export(self._current_results, output_path)

            return {
                'success': success,
                'path': output_path if success else None,
                'message': 'Export successful' if success else 'Export failed'
            }

        except Exception as e:
            logger.error(f"Export error: {e}")
            return {'success': False, 'message': str(e)}

    # === Settings Operations ===

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings"""
        return {
            'success': True,
            'settings': self.settings.to_dict()
        }

    def save_settings(self, settings_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save settings"""
        try:
            self.settings.update_from_dict(settings_data)
            return {'success': True, 'message': 'Settings saved'}
        except Exception as e:
            return {'success': False, 'message': str(e)}

    # === Single Image Processing (for testing) ===

    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image (for preview/testing)"""
        try:
            result = self.engine.process_image(image_path)

            if result.success and result.candidate:
                return {
                    'success': True,
                    'candidate': result.candidate.to_dict(),
                    'warnings': result.warnings,
                    'processing_time_ms': result.processing_time_ms
                }

            return {
                'success': False,
                'message': result.error_message or 'Processing failed'
            }

        except Exception as e:
            logger.error(f"Single image processing error: {e}")
            return {'success': False, 'message': str(e)}

    def select_single_image(self) -> Dict[str, Any]:
        """Select and process a single image"""
        try:
            result = self.window.create_file_dialog(
                webview.OPEN_DIALOG,
                file_types=(
                    'Image Files (*.jpg;*.jpeg;*.png;*.tiff;*.bmp)',
                )
            )

            if result and len(result) > 0:
                return self.process_single_image(result[0])

            return {'success': False, 'message': 'No file selected'}

        except Exception as e:
            return {'success': False, 'message': str(e)}

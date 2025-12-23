"""
Event handlers for GUI
"""
from typing import Callable, Dict, Any
from src.services.batch_processor import ProcessingProgress
from src.utils.logging_utils import get_logger

logger = get_logger('handlers')


class EventHandlers:
    """
    Event handlers for GUI interactions
    """

    def __init__(self):
        self._progress_listeners: list[Callable[[ProcessingProgress], None]] = []
        self._completion_listeners: list[Callable[[Dict[str, Any]], None]] = []

    def add_progress_listener(self, callback: Callable[[ProcessingProgress], None]) -> None:
        """Add listener for progress updates"""
        self._progress_listeners.append(callback)

    def add_completion_listener(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add listener for completion events"""
        self._completion_listeners.append(callback)

    def on_progress(self, progress: ProcessingProgress) -> None:
        """Handle progress update"""
        for listener in self._progress_listeners:
            try:
                listener(progress)
            except Exception as e:
                logger.error(f"Progress listener error: {e}")

    def on_completion(self, result: Dict[str, Any]) -> None:
        """Handle completion event"""
        for listener in self._completion_listeners:
            try:
                listener(result)
            except Exception as e:
                logger.error(f"Completion listener error: {e}")

    def remove_progress_listener(self, callback: Callable) -> None:
        """Remove progress listener"""
        if callback in self._progress_listeners:
            self._progress_listeners.remove(callback)

    def remove_completion_listener(self, callback: Callable) -> None:
        """Remove completion listener"""
        if callback in self._completion_listeners:
            self._completion_listeners.remove(callback)

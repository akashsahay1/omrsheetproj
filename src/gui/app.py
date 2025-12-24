"""
PyWebView application main module
"""
import os
import sys
import webview
from pathlib import Path
from typing import Optional

from src.gui.api import OMRApi
from src.config.settings import Settings
from src.config.constants import APP_TITLE
from src.utils.logging_utils import initialize_logging, get_logger

logger = get_logger('app')


class OMRApp:
    """
    Main application class for OMR Processor

    Manages the PyWebView window and API
    """

    def __init__(self):
        """Initialize the application"""
        # Ensure logging is initialized (may already be done in main.py)
        initialize_logging()

        # Load settings
        self.settings = Settings.load()

        # Initialize API
        self.api = OMRApi()

        # Window reference
        self.window: Optional[webview.Window] = None

    def get_ui_path(self) -> str:
        """Get path to UI files"""
        # Try different locations
        possible_paths = [
            # Development path
            Path(__file__).parent.parent.parent / 'ui' / 'index.html',
            # Installed path
            Path(sys.executable).parent / 'ui' / 'index.html',
            # Relative path
            Path('ui') / 'index.html',
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        # Fallback to bundled location (PyInstaller)
        if getattr(sys, 'frozen', False):
            base_path = Path(sys._MEIPASS)
            return str(base_path / 'ui' / 'index.html')

        raise FileNotFoundError("UI files not found")

    def run(self) -> None:
        """Run the application"""
        try:
            ui_path = self.get_ui_path()
            logger.info(f"Starting application with UI from: {ui_path}")

            # Create window
            self.window = webview.create_window(
                title=APP_TITLE,
                url=ui_path,
                js_api=self.api,
                width=self.settings.window_width,
                height=self.settings.window_height,
                min_size=(1024, 700),
                resizable=True,
                text_select=False
            )

            # Set window reference in API
            self.api.set_window(self.window)

            # Start webview
            webview.start(
                http_server=True,
                debug=False
            )

        except Exception as e:
            logger.error(f"Application startup error: {e}")
            raise


def main():
    """Entry point for the application"""
    app = OMRApp()
    app.run()


if __name__ == '__main__':
    main()

"""
Logging configuration utilities
"""
import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def get_log_directory() -> Path:
    """
    Get the log directory based on environment.

    Returns:
        Path to log directory:
        - Development: {PROJECT_ROOT}/logs/
        - Production (bundled): C:/Users/{username}/vbomr/
    """
    if getattr(sys, 'frozen', False):
        # Production: User's home directory
        log_dir = Path.home() / 'vbomr'
    else:
        # Development: Project root logs folder
        # Go up from src/utils to project root
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / 'logs'

    return log_dir


def get_log_file_path() -> Path:
    """
    Get the full path to the log file with date suffix.

    Returns:
        Path to omr-YYYY-MM-DD.log file
    """
    from datetime import date
    log_dir = get_log_directory()
    today = date.today().strftime('%Y-%m-%d')
    return log_dir / f'omr-{today}.log'


def ensure_log_directory() -> Path:
    """
    Create log directory if it doesn't exist.

    Returns:
        Path to log directory
    """
    log_dir = get_log_directory()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up application logging with file and console output.

    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file (uses default if not provided)
        console_output: Whether to output to console
        max_bytes: Maximum log file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)

    Returns:
        Root logger instance
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger
    root_logger = logging.getLogger('omr_processor')
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Determine log file path
    if log_file is None:
        # Use default log path based on environment
        log_path = get_log_file_path()
    else:
        log_path = Path(log_file)

    # Ensure log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Add rotating file handler
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Log the log file location on startup
        root_logger.info(f"Log file initialized at: {log_path}")
    except Exception as e:
        # If file logging fails, at least log to console
        root_logger.warning(f"Could not create log file at {log_path}: {e}")

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger.

    Args:
        name: Logger name (usually module name)

    Returns:
        Logger instance
    """
    return logging.getLogger(f'omr_processor.{name}')


# Default setup
_default_logger = None
_logging_initialized = False


def initialize_logging() -> logging.Logger:
    """
    Initialize logging system. Should be called once at application startup.

    Returns:
        Root logger instance
    """
    global _logging_initialized, _default_logger

    if not _logging_initialized:
        # Ensure log directory exists before anything else
        log_dir = ensure_log_directory()

        # Setup logging with file output
        _default_logger = setup_logging()

        _logging_initialized = True

        # Log startup info
        _default_logger.info("=" * 60)
        _default_logger.info("OMR Processor Application Starting")
        _default_logger.info(f"Log directory: {log_dir}")
        if getattr(sys, 'frozen', False):
            _default_logger.info("Running in production mode (bundled)")
        else:
            _default_logger.info("Running in development mode")
        _default_logger.info("=" * 60)

    return _default_logger


def get_default_logger() -> logging.Logger:
    """Get or create default logger"""
    global _default_logger
    if _default_logger is None:
        _default_logger = initialize_logging()
    return _default_logger

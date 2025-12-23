"""
File operation utilities
"""
import os
from pathlib import Path
from typing import List, Optional
from src.config.constants import SUPPORTED_IMAGE_FORMATS
from src.utils.logging_utils import get_logger

logger = get_logger('file_utils')


class FileUtils:
    """Utility class for file operations"""

    @staticmethod
    def get_image_files(folder_path: str) -> List[str]:
        """
        Get all supported image files from a folder

        Args:
            folder_path: Path to folder

        Returns:
            List of image file paths sorted by name
        """
        image_files = []

        try:
            folder = Path(folder_path)

            if not folder.exists():
                logger.warning(f"Folder does not exist: {folder_path}")
                return []

            if not folder.is_dir():
                logger.warning(f"Path is not a folder: {folder_path}")
                return []

            for file_path in folder.iterdir():
                if file_path.is_file():
                    ext = file_path.suffix.lower()
                    if ext in SUPPORTED_IMAGE_FORMATS:
                        image_files.append(str(file_path))

            # Sort by filename
            image_files.sort(key=lambda x: Path(x).name.lower())

            logger.info(f"Found {len(image_files)} image files in {folder_path}")
            return image_files

        except Exception as e:
            logger.error(f"Error scanning folder {folder_path}: {e}")
            return []

    @staticmethod
    def ensure_directory(path: str) -> bool:
        """
        Ensure directory exists, create if not

        Args:
            path: Directory path

        Returns:
            True if directory exists or was created
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}")
            return False

    @staticmethod
    def get_unique_filename(base_path: str) -> str:
        """
        Get unique filename by adding counter if file exists

        Args:
            base_path: Original file path

        Returns:
            Unique file path
        """
        path = Path(base_path)

        if not path.exists():
            return base_path

        stem = path.stem
        suffix = path.suffix
        parent = path.parent

        counter = 1
        while True:
            new_path = parent / f"{stem}_{counter}{suffix}"
            if not new_path.exists():
                return str(new_path)
            counter += 1

    @staticmethod
    def is_valid_image_file(file_path: str) -> bool:
        """
        Check if file is a valid image file

        Args:
            file_path: Path to file

        Returns:
            True if valid image file
        """
        path = Path(file_path)

        if not path.exists():
            return False

        if not path.is_file():
            return False

        ext = path.suffix.lower()
        return ext in SUPPORTED_IMAGE_FORMATS

    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Get file size in megabytes"""
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0

    @staticmethod
    def get_app_data_dir() -> str:
        """Get application data directory"""
        app_data = Path.home() / ".omr_processor"
        app_data.mkdir(exist_ok=True)
        return str(app_data)

    @staticmethod
    def get_temp_dir() -> str:
        """Get temporary directory for processing"""
        temp_dir = Path(FileUtils.get_app_data_dir()) / "temp"
        temp_dir.mkdir(exist_ok=True)
        return str(temp_dir)

    @staticmethod
    def clean_temp_dir() -> None:
        """Clean temporary directory"""
        temp_dir = Path(FileUtils.get_temp_dir())

        try:
            for file in temp_dir.iterdir():
                if file.is_file():
                    file.unlink()
        except Exception as e:
            logger.warning(f"Error cleaning temp directory: {e}")

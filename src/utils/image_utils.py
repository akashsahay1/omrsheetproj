"""
Image processing utilities
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from src.utils.logging_utils import get_logger

logger = get_logger('image_utils')


class ImageUtils:
    """Utility class for image operations"""

    @staticmethod
    def load_image(file_path: str) -> Optional[np.ndarray]:
        """
        Load an image from file

        Args:
            file_path: Path to image file

        Returns:
            Image as numpy array (BGR format) or None if failed
        """
        try:
            # Read image
            image = cv2.imread(file_path)

            if image is None:
                logger.error(f"Failed to load image: {file_path}")
                return None

            logger.debug(f"Loaded image: {file_path}, shape: {image.shape}")
            return image

        except Exception as e:
            logger.error(f"Error loading image {file_path}: {e}")
            return None

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Apply Gaussian blur for noise reduction"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                    tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        for better contrast
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    @staticmethod
    def apply_adaptive_threshold(image: np.ndarray,
                                  block_size: int = 11,
                                  c: int = 2) -> np.ndarray:
        """Apply adaptive thresholding"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return cv2.adaptiveThreshold(
            image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size, c
        )

    @staticmethod
    def apply_otsu_threshold(image: np.ndarray) -> np.ndarray:
        """Apply Otsu's thresholding"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return thresh

    @staticmethod
    def resize_image(image: np.ndarray,
                     width: Optional[int] = None,
                     height: Optional[int] = None,
                     scale: Optional[float] = None) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio

        Args:
            image: Input image
            width: Target width (optional)
            height: Target height (optional)
            scale: Scale factor (optional)

        Returns:
            Resized image
        """
        h, w = image.shape[:2]

        if scale is not None:
            new_width = int(w * scale)
            new_height = int(h * scale)
        elif width is not None and height is not None:
            new_width = width
            new_height = height
        elif width is not None:
            ratio = width / w
            new_width = width
            new_height = int(h * ratio)
        elif height is not None:
            ratio = height / h
            new_width = int(w * ratio)
            new_height = height
        else:
            return image

        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def extract_roi(image: np.ndarray,
                    x: int, y: int,
                    width: int, height: int) -> np.ndarray:
        """Extract region of interest from image"""
        h, w = image.shape[:2]

        # Clamp coordinates
        x1 = max(0, min(x, w - 1))
        y1 = max(0, min(y, h - 1))
        x2 = max(0, min(x + width, w))
        y2 = max(0, min(y + height, h))

        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def calculate_mean_intensity(image: np.ndarray) -> float:
        """Calculate mean pixel intensity of image"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(image))

    @staticmethod
    def calculate_fill_ratio(image: np.ndarray, threshold: int = 128) -> float:
        """
        Calculate ratio of dark pixels (potential marks)

        Args:
            image: Grayscale image
            threshold: Intensity threshold for dark pixels

        Returns:
            Ratio of dark pixels (0.0 to 1.0)
        """
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        total_pixels = image.size
        dark_pixels = np.sum(image < threshold)

        return dark_pixels / total_pixels if total_pixels > 0 else 0.0

    @staticmethod
    def find_contours(image: np.ndarray) -> List:
        """Find contours in binary image"""
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    @staticmethod
    def get_perspective_transform(
        src_points: np.ndarray,
        dst_points: np.ndarray,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Apply perspective transformation

        Args:
            src_points: Source points (4 corners)
            dst_points: Destination points (4 corners)
            image: Input image

        Returns:
            Warped image
        """
        matrix = cv2.getPerspectiveTransform(
            src_points.astype(np.float32),
            dst_points.astype(np.float32)
        )

        h, w = image.shape[:2]
        return cv2.warpPerspective(image, matrix, (w, h))

    @staticmethod
    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image by given angle

        Args:
            image: Input image
            angle: Rotation angle in degrees

        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def detect_skew_angle(image: np.ndarray) -> float:
        """
        Detect skew angle of document

        Args:
            image: Input image (grayscale or BGR)

        Returns:
            Skew angle in degrees
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is None:
            return 0.0

        # Calculate angles
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = np.degrees(theta) - 90
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return 0.0

        # Return median angle
        return float(np.median(angles))

    @staticmethod
    def deskew_image(image: np.ndarray) -> np.ndarray:
        """
        Automatically deskew document image

        Args:
            image: Input image

        Returns:
            Deskewed image
        """
        angle = ImageUtils.detect_skew_angle(image)

        if abs(angle) > 0.5:  # Only correct if skew is significant
            return ImageUtils.rotate_image(image, angle)

        return image

    @staticmethod
    def save_image(image: np.ndarray, file_path: str) -> bool:
        """Save image to file"""
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(file_path, image)
            return True
        except Exception as e:
            logger.error(f"Error saving image {file_path}: {e}")
            return False

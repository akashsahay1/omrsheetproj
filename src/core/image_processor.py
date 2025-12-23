"""
Image preprocessing for OMR sheets
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from src.utils.image_utils import ImageUtils
from src.utils.logging_utils import get_logger

logger = get_logger('image_processor')


class ImageProcessor:
    """Handles image preprocessing for OMR detection"""

    def __init__(self,
                 target_width: int = 2480,
                 target_height: int = 3507):
        """
        Initialize image processor

        Args:
            target_width: Target image width
            target_height: Target image height
        """
        self.target_width = target_width
        self.target_height = target_height

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline

        Args:
            image: Input image (BGR)

        Returns:
            Tuple of (processed_grayscale, original_resized)
        """
        # Resize to target dimensions
        resized = self._resize_to_target(image)

        # Convert to grayscale
        gray = ImageUtils.to_grayscale(resized)

        # Apply Gaussian blur
        blurred = ImageUtils.apply_gaussian_blur(gray, kernel_size=3)

        # Apply CLAHE for contrast enhancement
        enhanced = ImageUtils.apply_clahe(blurred)

        logger.debug(f"Preprocessed image: {image.shape} -> {enhanced.shape}")

        return enhanced, resized

    def _resize_to_target(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions"""
        h, w = image.shape[:2]

        # Check if resize is needed
        if w == self.target_width and h == self.target_height:
            return image

        # Resize maintaining aspect ratio
        aspect = w / h
        target_aspect = self.target_width / self.target_height

        if abs(aspect - target_aspect) < 0.1:
            # Similar aspect ratio, direct resize
            return cv2.resize(image, (self.target_width, self.target_height),
                             interpolation=cv2.INTER_AREA)
        else:
            # Different aspect, resize and pad
            if aspect > target_aspect:
                # Image is wider, fit to width
                new_width = self.target_width
                new_height = int(self.target_width / aspect)
            else:
                # Image is taller, fit to height
                new_height = self.target_height
                new_width = int(self.target_height * aspect)

            resized = cv2.resize(image, (new_width, new_height),
                                interpolation=cv2.INTER_AREA)

            # Pad to target size
            pad_h = self.target_height - new_height
            pad_w = self.target_width - new_width

            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=[255, 255, 255])

            return padded

    def detect_page_boundaries(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect page boundaries in image

        Args:
            image: Grayscale image

        Returns:
            4 corner points as numpy array, or None if not found
        """
        # Apply threshold
        _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological operations
        kernel = np.ones((10, 10), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour (should be page boundary)
        largest = max(contours, key=cv2.contourArea)

        # Approximate to quadrilateral
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

        # If not exactly 4 points, use bounding rectangle
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return np.int0(box)

    def crop_to_page(self, image: np.ndarray,
                     corners: np.ndarray) -> np.ndarray:
        """
        Crop and warp image to page boundaries

        Args:
            image: Input image
            corners: 4 corner points

        Returns:
            Cropped and perspective-corrected image
        """
        # Order points: top-left, top-right, bottom-right, bottom-left
        corners = self._order_points(corners)

        # Define destination points
        dst_points = np.array([
            [0, 0],
            [self.target_width - 1, 0],
            [self.target_width - 1, self.target_height - 1],
            [0, self.target_height - 1]
        ], dtype=np.float32)

        # Get perspective transform
        matrix = cv2.getPerspectiveTransform(
            corners.astype(np.float32),
            dst_points
        )

        # Apply warp
        warped = cv2.warpPerspective(
            image, matrix,
            (self.target_width, self.target_height)
        )

        return warped

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in: top-left, top-right, bottom-right, bottom-left

        Args:
            pts: Unordered 4 points

        Returns:
            Ordered points
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        # Sum of coordinates: smallest = top-left, largest = bottom-right
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # Difference: smallest = top-right, largest = bottom-left
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Correct skew in document image

        Args:
            image: Input image

        Returns:
            Deskewed image
        """
        return ImageUtils.deskew_image(image)

    def enhance_for_detection(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Enhance grayscale image for bubble detection

        Args:
            gray_image: Grayscale image

        Returns:
            Enhanced image
        """
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray_image)

        # Slight blur to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return enhanced

    def get_binary_mask(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Get binary mask for bubble detection

        Args:
            gray_image: Grayscale image

        Returns:
            Binary image
        """
        return ImageUtils.apply_adaptive_threshold(
            gray_image, block_size=15, c=5
        )

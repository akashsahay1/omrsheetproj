"""
Sheet alignment and marker detection
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
from src.utils.image_utils import ImageUtils
from src.utils.logging_utils import get_logger

logger = get_logger('alignment')


class AlignmentDetector:
    """Detects and corrects sheet alignment"""

    def __init__(self):
        self.confidence = 0.0

    def detect_markers(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect alignment markers/corners on the sheet

        Args:
            image: Grayscale image

        Returns:
            4 corner points or None if detection failed
        """
        # Try multiple detection methods
        corners = self._detect_by_contours(image)

        if corners is not None:
            self.confidence = 0.9
            return corners

        corners = self._detect_by_lines(image)

        if corners is not None:
            self.confidence = 0.7
            return corners

        logger.warning("Marker detection failed")
        self.confidence = 0.0
        return None

    def _detect_by_contours(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect page using contour detection"""
        # Apply threshold
        _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)

        # Check if large enough (at least 50% of image area)
        h, w = image.shape[:2]
        area = cv2.contourArea(largest)
        if area < 0.5 * h * w:
            return None

        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        if len(approx) == 4:
            return approx.reshape(4, 2)

        # Use minimum area rectangle
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return np.int0(box)

    def _detect_by_lines(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect page boundaries using line detection"""
        # Edge detection
        edges = cv2.Canny(image, 50, 150)

        # Detect lines
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 100,
            minLineLength=100, maxLineGap=10
        )

        if lines is None:
            return None

        # Find horizontal and vertical lines
        h_lines = []
        v_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)

            if angle < 10 or angle > 170:
                h_lines.append(line[0])
            elif 80 < angle < 100:
                v_lines.append(line[0])

        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        # Find extreme lines
        h, w = image.shape[:2]

        top_line = min(h_lines, key=lambda l: (l[1] + l[3])/2)
        bottom_line = max(h_lines, key=lambda l: (l[1] + l[3])/2)
        left_line = min(v_lines, key=lambda l: (l[0] + l[2])/2)
        right_line = max(v_lines, key=lambda l: (l[0] + l[2])/2)

        # Calculate intersection points
        corners = np.array([
            self._line_intersection(top_line, left_line),
            self._line_intersection(top_line, right_line),
            self._line_intersection(bottom_line, right_line),
            self._line_intersection(bottom_line, left_line)
        ])

        return corners if all(c is not None for c in corners) else None

    def _line_intersection(self, line1, line2) -> Optional[Tuple[int, int]]:
        """Calculate intersection of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)

        if abs(denom) < 1e-10:
            return None

        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom

        x = x1 + t*(x2-x1)
        y = y1 + t*(y2-y1)

        return (int(x), int(y))

    def align_sheet(self,
                    image: np.ndarray,
                    corners: np.ndarray,
                    target_width: int,
                    target_height: int) -> np.ndarray:
        """
        Warp image to correct alignment

        Args:
            image: Input image
            corners: Detected corner points
            target_width: Target width
            target_height: Target height

        Returns:
            Aligned image
        """
        # Order corners
        ordered = self._order_corners(corners)

        # Target points
        dst = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)

        # Perspective transform
        matrix = cv2.getPerspectiveTransform(
            ordered.astype(np.float32), dst
        )

        warped = cv2.warpPerspective(
            image, matrix, (target_width, target_height)
        )

        return warped

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """Order corners: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]
        rect[2] = corners[np.argmax(s)]

        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]
        rect[3] = corners[np.argmax(diff)]

        return rect

    def auto_rotate(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Automatically correct rotation

        Args:
            image: Input image

        Returns:
            Tuple of (rotated_image, angle)
        """
        angle = ImageUtils.detect_skew_angle(image)

        if abs(angle) > 5.0:  # Higher threshold to avoid breaking coordinate-based detection
            rotated = ImageUtils.rotate_image(image, angle)
            logger.info(f"Auto-rotated by {angle:.2f} degrees")
            return rotated, angle

        return image, 0.0

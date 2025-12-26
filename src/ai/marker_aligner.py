"""
Marker-based sheet alignment module.

Detects corner markers (black squares/circles) on the OMR sheet and
performs perspective transformation to normalize the image.

This removes dependency on scanner positioning and handles:
- Rotation
- Skew
- Scale differences
- Translation
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class AlignmentResult:
    """Result of sheet alignment."""
    success: bool
    aligned_image: Optional[np.ndarray] = None
    corners_detected: int = 0
    confidence: float = 0.0
    error_message: str = ""


class MarkerAligner:
    """
    Aligns OMR sheets using corner marker detection.

    The OMR sheet should have distinctive markers (black squares or circles)
    at the corners that can be detected and used for perspective correction.
    """

    # Standard output dimensions (A4 at 300 DPI)
    OUTPUT_WIDTH = 2480
    OUTPUT_HEIGHT = 3507

    def __init__(
        self,
        marker_template_path: Optional[str] = None,
        min_marker_area: int = 500,
        max_marker_area: int = 10000
    ):
        """
        Initialize the marker aligner.

        Args:
            marker_template_path: Path to marker template image for matching
            min_marker_area: Minimum area for a detected marker
            max_marker_area: Maximum area for a detected marker
        """
        self.marker_template = None
        if marker_template_path:
            self.marker_template = cv2.imread(marker_template_path, cv2.IMREAD_GRAYSCALE)

        self.min_marker_area = min_marker_area
        self.max_marker_area = max_marker_area

    def align(self, image: np.ndarray) -> AlignmentResult:
        """
        Align the OMR sheet image using detected corner markers.

        Args:
            image: Input BGR image

        Returns:
            AlignmentResult with aligned image if successful
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Detect corners using multiple methods
            corners = self._detect_corners(gray)

            if len(corners) < 4:
                # Try alternative detection
                corners = self._detect_corners_contour(gray)

            if len(corners) < 4:
                # Fall back to edge-based detection
                corners = self._detect_corners_edge(gray)

            if len(corners) < 4:
                return AlignmentResult(
                    success=False,
                    corners_detected=len(corners),
                    error_message=f"Only {len(corners)} corners detected, need 4"
                )

            # Order corners: top-left, top-right, bottom-right, bottom-left
            ordered_corners = self._order_corners(corners[:4])

            # Perform perspective transform
            aligned = self._perspective_transform(image, ordered_corners)

            return AlignmentResult(
                success=True,
                aligned_image=aligned,
                corners_detected=4,
                confidence=0.95
            )

        except Exception as e:
            return AlignmentResult(
                success=False,
                error_message=str(e)
            )

    def _detect_corners(self, gray: np.ndarray) -> List[Tuple[int, int]]:
        """Detect corner markers using template matching or blob detection."""
        corners = []

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 10
        )

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours by area and shape
        marker_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_marker_area < area < self.max_marker_area:
                # Check if roughly square or circular
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

                # Squares have circularity ~0.785, circles ~1.0
                if circularity > 0.5:
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        marker_candidates.append((cx, cy, area))

        # Select the 4 markers closest to the corners
        h, w = gray.shape
        corner_regions = [
            (0, 0),           # Top-left
            (w, 0),           # Top-right
            (w, h),           # Bottom-right
            (0, h)            # Bottom-left
        ]

        for corner_x, corner_y in corner_regions:
            best_marker = None
            best_dist = float('inf')

            for mx, my, area in marker_candidates:
                dist = np.sqrt((mx - corner_x)**2 + (my - corner_y)**2)
                if dist < best_dist and dist < max(w, h) * 0.3:  # Within 30% of image
                    best_dist = dist
                    best_marker = (mx, my)

            if best_marker:
                corners.append(best_marker)

        return corners

    def _detect_corners_contour(self, gray: np.ndarray) -> List[Tuple[int, int]]:
        """Detect page corners using contour detection."""
        corners = []

        # Blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return corners

        # Find the largest contour (should be the page)
        largest = max(contours, key=cv2.contourArea)

        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        if len(approx) >= 4:
            # Get the 4 corners
            points = approx.reshape(-1, 2)
            corners = [(int(p[0]), int(p[1])) for p in points[:4]]

        return corners

    def _detect_corners_edge(self, gray: np.ndarray) -> List[Tuple[int, int]]:
        """Detect corners using Harris corner detection."""
        corners = []

        # Harris corner detection
        gray_float = np.float32(gray)
        dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)

        # Threshold
        threshold = 0.01 * dst.max()
        corner_points = np.where(dst > threshold)

        if len(corner_points[0]) == 0:
            return corners

        # Cluster corners to find the 4 main ones
        points = list(zip(corner_points[1], corner_points[0]))

        h, w = gray.shape
        corner_regions = [
            (0, 0, w//4, h//4),                    # Top-left
            (3*w//4, 0, w, h//4),                  # Top-right
            (3*w//4, 3*h//4, w, h),                # Bottom-right
            (0, 3*h//4, w//4, h)                   # Bottom-left
        ]

        for x1, y1, x2, y2 in corner_regions:
            region_points = [(x, y) for x, y in points if x1 <= x < x2 and y1 <= y < y2]
            if region_points:
                # Average the points in this region
                avg_x = int(np.mean([p[0] for p in region_points]))
                avg_y = int(np.mean([p[1] for p in region_points]))
                corners.append((avg_x, avg_y))

        return corners

    def _order_corners(self, corners: List[Tuple[int, int]]) -> np.ndarray:
        """Order corners as: top-left, top-right, bottom-right, bottom-left."""
        pts = np.array(corners, dtype=np.float32)

        # Sort by sum (x+y) - smallest is top-left, largest is bottom-right
        s = pts.sum(axis=1)
        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]

        # Sort by difference (y-x) - smallest is top-right, largest is bottom-left
        d = np.diff(pts, axis=1)
        top_right = pts[np.argmin(d)]
        bottom_left = pts[np.argmax(d)]

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

    def _perspective_transform(
        self,
        image: np.ndarray,
        corners: np.ndarray
    ) -> np.ndarray:
        """Apply perspective transformation to normalize the image."""
        # Destination points (normalized rectangle)
        dst = np.array([
            [0, 0],
            [self.OUTPUT_WIDTH - 1, 0],
            [self.OUTPUT_WIDTH - 1, self.OUTPUT_HEIGHT - 1],
            [0, self.OUTPUT_HEIGHT - 1]
        ], dtype=np.float32)

        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(corners, dst)

        # Apply transform
        aligned = cv2.warpPerspective(
            image, M, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT)
        )

        return aligned

    def align_with_reference(
        self,
        image: np.ndarray,
        reference: np.ndarray
    ) -> AlignmentResult:
        """
        Align image using feature matching with a reference image.

        This is more robust than marker detection for sheets without
        clear corner markers.
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            if len(reference.shape) == 3:
                ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = reference.copy()

            # Detect ORB features
            orb = cv2.ORB_create(2000)
            kp1, des1 = orb.detectAndCompute(gray, None)
            kp2, des2 = orb.detectAndCompute(ref_gray, None)

            if des1 is None or des2 is None:
                return AlignmentResult(
                    success=False,
                    error_message="Could not detect features"
                )

            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Use top matches
            good_matches = matches[:int(len(matches) * 0.15)]

            if len(good_matches) < 4:
                return AlignmentResult(
                    success=False,
                    error_message="Not enough good matches"
                )

            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                return AlignmentResult(
                    success=False,
                    error_message="Could not compute homography"
                )

            # Warp image
            h, w = ref_gray.shape
            aligned = cv2.warpPerspective(image, H, (w, h))

            return AlignmentResult(
                success=True,
                aligned_image=aligned,
                corners_detected=4,
                confidence=0.9
            )

        except Exception as e:
            return AlignmentResult(
                success=False,
                error_message=str(e)
            )

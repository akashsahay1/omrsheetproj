"""
Marker-based sheet alignment module.

Detects corner markers (black squares/circles) on the OMR sheet and
performs perspective transformation to normalize the image.

This removes dependency on scanner positioning and handles:
- Rotation
- Skew
- Scale differences
- Translation

Key improvement: Uses canonical marker positions specific to Vidya Bharti OMR sheets
so all aligned images have markers at identical positions regardless of original dimensions.
"""

import itertools

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
    markers: Optional[np.ndarray] = None  # Detected marker positions
    original_size: Tuple[int, int] = (0, 0)  # (width, height) of original image


class MarkerAligner:
    """
    Aligns OMR sheets using corner marker detection.

    The OMR sheet should have distinctive markers (black squares or circles)
    at the corners that can be detected and used for perspective correction.

    After alignment, all images will have markers at CANONICAL_MARKERS positions
    in a TARGET_WIDTH x TARGET_HEIGHT frame, enabling consistent coordinate-based
    processing regardless of original image dimensions or scan quality.
    """

    # Standard output dimensions (A4 at 300 DPI)
    OUTPUT_WIDTH = 2480
    OUTPUT_HEIGHT = 3507

    # Canonical marker positions in the aligned output
    # These are where the markers should end up after perspective transform
    CANONICAL_MARKERS = np.float32([
        [170, 530],    # Top-left marker center
        [2310, 530],   # Top-right marker center
        [170, 3150],   # Bottom-left marker center
        [2310, 3150]   # Bottom-right marker center
    ])

    # Candidates kept per corner when searching for a valid marker rectangle
    CANDIDATES_PER_CORNER = 4

    def __init__(
        self,
        marker_template_path: Optional[str] = None,
        min_marker_area: int = 300,
        max_marker_area: int = 25000  # Increased to handle merged markers
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

        Uses perspective transform to warp detected markers to CANONICAL_MARKERS
        positions, ensuring all aligned images have consistent coordinates.

        Args:
            image: Input BGR image

        Returns:
            AlignmentResult with aligned image if successful
        """
        h, w = image.shape[:2]
        original_size = (w, h)

        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Detect corners using multiple methods
            corners = self._detect_corners(gray)

            if len(corners) == 3:
                # Try to estimate missing corner from 3 detected ones first
                # This is more reliable than contour/edge detection
                estimated = self._estimate_missing_corners(corners, w, h)
                if len(estimated) == 4:
                    corners = estimated

            if len(corners) < 4:
                # Try alternative detection
                corners = self._detect_corners_contour(gray)

            if len(corners) < 4:
                # Fall back to edge-based detection
                corners = self._detect_corners_edge(gray)

            if len(corners) < 4:
                # Try to estimate missing corners from detected ones
                corners = self._estimate_missing_corners(corners, w, h)

            if len(corners) < 4:
                # Final fallback: simple resize without perspective correction
                aligned = cv2.resize(image, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))
                return AlignmentResult(
                    success=False,
                    aligned_image=aligned,
                    corners_detected=len(corners),
                    error_message=f"Only {len(corners)} corners detected, using simple resize fallback",
                    original_size=original_size
                )

            # Order corners: top-left, top-right, bottom-right, bottom-left
            ordered_corners = self._order_corners(corners[:4])

            # Perform perspective transform to canonical positions
            aligned = self._perspective_transform(image, ordered_corners)

            # Fallback detectors (page contour / Harris) can lock onto a printed
            # box instead of the markers; don't claim success for those.
            tl, tr, bl, br = ordered_corners
            if not self._is_rectangular(tl, tr, bl, br):
                return AlignmentResult(
                    success=False,
                    aligned_image=aligned,
                    corners_detected=4,
                    confidence=0.3,
                    error_message="Detected corners do not form a rectangle",
                    markers=ordered_corners,
                    original_size=original_size
                )

            return AlignmentResult(
                success=True,
                aligned_image=aligned,
                corners_detected=4,
                confidence=0.95,
                markers=ordered_corners,
                original_size=original_size
            )

        except Exception as e:
            # Fallback to simple resize on any error
            aligned = cv2.resize(image, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))
            return AlignmentResult(
                success=False,
                aligned_image=aligned,
                error_message=str(e),
                original_size=original_size
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

        # Area limits are tuned for ~1650px-wide scans; scale for other DPIs
        area_scale = (gray.shape[1] / 1656.0) ** 2
        min_area = self.min_marker_area * area_scale
        max_area = self.max_marker_area * area_scale

        # Filter contours by area and shape
        marker_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
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

        # Filter out abnormally large markers (likely logos/text, not actual markers)
        # Keep only markers with area < 3x the median area
        if len(marker_candidates) >= 3:
            areas = sorted([m[2] for m in marker_candidates])
            median_area = areas[len(areas) // 2]
            max_allowed = median_area * 5  # Allow up to 5x median
            marker_candidates = [(x, y, a) for x, y, a in marker_candidates if a < max_allowed]

        # Select the 4 markers closest to the corners
        h, w = gray.shape
        corner_regions = [
            (0, 0),           # Top-left
            (w, 0),           # Top-right
            (w, h),           # Bottom-right
            (0, h)            # Bottom-left
        ]

        # Typical marker area range (Vidya Bharti forms have markers ~500-700 pixels
        # at ~1650px width). Scale with image width so other scan DPIs work too.
        area_scale = (w / 1656.0) ** 2
        TYPICAL_AREA_MIN = 400 * area_scale
        TYPICAL_AREA_MAX = 1500 * area_scale

        # Keep the best few candidates per corner, not just the closest: a round
        # blob of a school stamp near a corner can sit closer to the page corner
        # than the real marker.
        per_corner = []
        for corner_x, corner_y in corner_regions:
            scored = []
            for mx, my, area in marker_candidates:
                dist = np.sqrt((mx - corner_x)**2 + (my - corner_y)**2)
                # Strict corner region: within 20% of image dimension
                if dist < max(w, h) * 0.20:
                    # Prefer markers with typical area over abnormally large ones
                    # Score = distance + penalty for non-typical area
                    if TYPICAL_AREA_MIN <= area <= TYPICAL_AREA_MAX:
                        area_penalty = 0
                    else:
                        # Large penalty for markers outside typical range
                        area_penalty = abs(area - (TYPICAL_AREA_MIN + TYPICAL_AREA_MAX) / 2) * 0.5
                    scored.append((dist + area_penalty, (mx, my)))
            scored.sort(key=lambda s: s[0])
            per_corner.append(scored[:self.CANDIDATES_PER_CORNER])

        if not all(per_corner):
            # Some corner has no candidate: return what we have so the caller
            # can estimate the missing corner from the other three
            return [c[0][1] for c in per_corner if c]

        # Try combinations best-score first; use the first one that forms a
        # proper page rectangle (a stamp blob won't line up with the others)
        combos = sorted(itertools.product(*per_corner), key=lambda combo: sum(s for s, _ in combo))
        for combo in combos:
            corners = [pt for _, pt in combo]
            if self._validate_corners(corners, w, h):
                return corners

        # No valid rectangle: return empty to trigger fallback
        return []

    def _validate_corners(self, corners: List[Tuple[int, int]], w: int, h: int) -> bool:
        """Validate that detected corners form a reasonable quadrilateral."""
        if len(corners) != 4:
            return False

        # Order corners for validation
        pts = np.array(corners, dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).flatten()

        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]

        # Check 1: TL should be in top-left quadrant (strict: within 15% of corner)
        if tl[0] > w * 0.15 or tl[1] > h * 0.25:
            return False

        # Check 2: BR should be in bottom-right quadrant
        if br[0] < w * 0.85 or br[1] < h * 0.75:
            return False

        # Check 3: TR should be right side, top half
        if tr[0] < w * 0.85 or tr[1] > h * 0.25:
            return False

        # Check 4: BL should be left side, bottom half
        if bl[0] > w * 0.15 or bl[1] < h * 0.75:
            return False

        # Check 5: Width and height should be reasonable. Scanners/printers vary
        # the margins and page scale throughout the year (e.g. markers span 73%
        # of the height on one batch and 69% on another), so keep this loose and
        # rely on the geometric checks below instead.
        width = np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
        height = np.sqrt((bl[0] - tl[0])**2 + (bl[1] - tl[1])**2)

        if width < w * 0.6 or height < h * 0.5:
            return False

        return self._is_rectangular(tl, tr, bl, br)

    @staticmethod
    def _is_rectangular(tl, tr, bl, br) -> bool:
        """Check that four marker points form a near-rectangle of plausible aspect."""
        top = np.linalg.norm(tr - tl)
        bottom = np.linalg.norm(br - bl)
        left = np.linalg.norm(bl - tl)
        right = np.linalg.norm(br - tr)
        if min(top, bottom, left, right) <= 0:
            return False

        # Opposite sides should match within 5%
        if abs(top - bottom) / max(top, bottom) > 0.05:
            return False
        if abs(left - right) / max(left, right) > 0.05:
            return False

        # Diagonals should match within 5% (rules out parallelograms)
        d1 = np.linalg.norm(br - tl)
        d2 = np.linalg.norm(bl - tr)
        if abs(d1 - d2) / max(d1, d2) > 0.05:
            return False

        # Marker box on Vidya Bharti sheets is portrait, width/height ~0.85-0.95
        aspect = (top + bottom) / (left + right)
        return 0.7 < aspect < 1.1

    def _estimate_missing_corners(self, corners: List[Tuple[int, int]], w: int, h: int) -> List[Tuple[int, int]]:
        """Estimate missing corners from detected ones, assuming rectangular form."""
        if len(corners) >= 4:
            return corners

        if len(corners) < 3:
            # Need at least 3 corners to estimate the 4th
            return corners

        # Identify which corners we have based on position
        corner_regions = {
            'TL': (0, w * 0.3, 0, h * 0.3),
            'TR': (w * 0.7, w, 0, h * 0.3),
            'BL': (0, w * 0.3, h * 0.7, h),
            'BR': (w * 0.7, w, h * 0.7, h)
        }

        detected = {'TL': None, 'TR': None, 'BL': None, 'BR': None}
        for cx, cy in corners:
            for name, (x1, x2, y1, y2) in corner_regions.items():
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    detected[name] = (cx, cy)
                    break

        # Count detected corners
        found = [k for k, v in detected.items() if v is not None]
        missing = [k for k, v in detected.items() if v is None]

        if len(found) == 3 and len(missing) == 1:
            # Estimate the missing corner
            miss = missing[0]

            if miss == 'TL' and detected['TR'] and detected['BL'] and detected['BR']:
                # TL = TR + BL - BR (parallelogram rule)
                tr, bl, br = detected['TR'], detected['BL'], detected['BR']
                tl = (tr[0] + bl[0] - br[0], tr[1] + bl[1] - br[1])
                detected['TL'] = tl

            elif miss == 'TR' and detected['TL'] and detected['BL'] and detected['BR']:
                tl, bl, br = detected['TL'], detected['BL'], detected['BR']
                tr = (tl[0] + br[0] - bl[0], tl[1] + br[1] - bl[1])
                detected['TR'] = tr

            elif miss == 'BL' and detected['TL'] and detected['TR'] and detected['BR']:
                tl, tr, br = detected['TL'], detected['TR'], detected['BR']
                bl = (tl[0] + br[0] - tr[0], tl[1] + br[1] - tr[1])
                detected['BL'] = bl

            elif miss == 'BR' and detected['TL'] and detected['TR'] and detected['BL']:
                tl, tr, bl = detected['TL'], detected['TR'], detected['BL']
                br = (tr[0] + bl[0] - tl[0], tr[1] + bl[1] - tl[1])
                detected['BR'] = br

        # Return all detected corners
        result = [v for v in detected.values() if v is not None]
        return result

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
        """Order corners as: top-left, top-right, bottom-left, bottom-right.

        This matches the CANONICAL_MARKERS order for perspective transform.
        """
        pts = np.array(corners, dtype=np.float32)

        # Sort by sum (x+y) - smallest is top-left, largest is bottom-right
        s = pts.sum(axis=1)
        top_left = pts[np.argmin(s)]
        bottom_right = pts[np.argmax(s)]

        # Sort by difference (y-x) - smallest is top-right, largest is bottom-left
        d = np.diff(pts, axis=1).flatten()
        top_right = pts[np.argmin(d)]
        bottom_left = pts[np.argmax(d)]

        # Order: TL, TR, BL, BR (matches CANONICAL_MARKERS)
        return np.array([top_left, top_right, bottom_left, bottom_right], dtype=np.float32)

    def _perspective_transform(
        self,
        image: np.ndarray,
        corners: np.ndarray
    ) -> np.ndarray:
        """Apply perspective transformation to canonical marker positions.

        Transforms detected markers to CANONICAL_MARKERS positions, ensuring
        all aligned images have markers at identical coordinates.
        """
        # Destination points are the canonical marker positions
        dst = self.CANONICAL_MARKERS

        # Compute perspective transform matrix
        M = cv2.getPerspectiveTransform(corners, dst)

        # Apply transform with white border for out-of-bounds regions
        aligned = cv2.warpPerspective(
            image, M, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
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


def align_image(image: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Convenience function to align an OMR image.

    Args:
        image: Input image (BGR or grayscale)

    Returns:
        (aligned_image, success) tuple
    """
    aligner = MarkerAligner()
    result = aligner.align(image)
    return result.aligned_image, result.success


def get_canonical_markers() -> np.ndarray:
    """Get the canonical marker positions used for alignment."""
    return MarkerAligner.CANONICAL_MARKERS.copy()

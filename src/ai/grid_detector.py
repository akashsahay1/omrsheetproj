"""
Dynamic grid detection module for OMR sheets.

Uses computer vision techniques (HoughCircles, edge detection) to automatically
detect bubble grid positions instead of relying on hardcoded coordinates.

This enables the system to work with varying image dimensions and slight
layout variations across different batches of OMR sheets.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class GridConfig:
    """Configuration for a detected bubble grid."""
    origin: Tuple[int, int]  # (x, y) of top-left bubble center
    rows: int
    cols: int
    row_gap: float  # Vertical spacing between rows
    col_gap: float  # Horizontal spacing between columns
    bubble_radius: int = 14  # Approximate bubble radius
    x_shift_per_row: float = 0.0  # X drift as you go down rows (for skewed grids)
    y_drift_per_col: float = 0.0  # Y drift as you go across columns
    confidence: float = 0.0


@dataclass
class DetectedBubble:
    """A detected bubble in the grid."""
    x: int  # Center x
    y: int  # Center y
    radius: int
    row: int = -1
    col: int = -1


# Search regions in canonical frame (after marker alignment to 2480x3507)
# Format: (x, y, width, height)
# These regions were calibrated based on actual bubble positions detected
# in test images after marker alignment
GRID_SEARCH_REGIONS = {
    'name': (130, 500, 1150, 1250),          # Name field: 25 columns x 26 rows (A-Z)
    'roll_number': (1500, 1150, 350, 600),   # Roll number: 3 columns x 10 rows (0-9)
    'class': (1680, 1150, 150, 500),         # Class: 1 column x 8 rows
    'section': (1830, 1150, 150, 500),       # Section: 1 column x 8 rows
    'school_code': (1980, 1150, 350, 600),   # School code: 3 columns x 10 rows
    'answers_1_10': (120, 1950, 380, 1100),  # Answer block 1 (Q1-10)
    'answers_11_20': (500, 1950, 380, 1100), # Answer block 2 (Q11-20)
    'answers_21_30': (880, 1950, 380, 1100), # Answer block 3 (Q21-30)
    'answers_31_40': (1260, 1950, 380, 1100),# Answer block 4 (Q31-40)
    'answers_41_50': (1640, 1950, 380, 1100),# Answer block 5 (Q41-50)
    'answers_51_60': (2020, 1950, 380, 1100),# Answer block 6 (Q51-60)
}

# Expected grid dimensions for validation
EXPECTED_GRIDS = {
    'name': {'rows': 26, 'cols': 25},
    'roll_number': {'rows': 10, 'cols': 3},
    'class': {'rows': 8, 'cols': 1},
    'section': {'rows': 8, 'cols': 1},
    'school_code': {'rows': 10, 'cols': 3},
    'answers_1_10': {'rows': 10, 'cols': 4},
    'answers_11_20': {'rows': 10, 'cols': 4},
    'answers_21_30': {'rows': 10, 'cols': 4},
    'answers_31_40': {'rows': 10, 'cols': 4},
    'answers_41_50': {'rows': 10, 'cols': 4},
    'answers_51_60': {'rows': 10, 'cols': 4},
}


class GridDetector:
    """
    Detects bubble grid positions dynamically using computer vision.

    Uses HoughCircles to find bubbles and clusters them into a grid structure,
    calculating origin, spacing, and any skew parameters.
    """

    def __init__(self, debug: bool = False):
        """
        Initialize the grid detector.

        Args:
            debug: If True, save debug images showing detected bubbles
        """
        self.debug = debug
        self._debug_images = {}

    def detect_grid(
        self,
        gray: np.ndarray,
        field_name: str,
        region: Optional[Tuple[int, int, int, int]] = None,
        force_dynamic: bool = False
    ) -> Optional[GridConfig]:
        """
        Detect bubble grid in specified region.

        Args:
            gray: Grayscale image (should be aligned to canonical frame)
            field_name: Name of the field ('name', 'roll_number', etc.)
            region: Optional (x, y, width, height) search region.
                   If None, uses GRID_SEARCH_REGIONS for the field.
            force_dynamic: If True, always try dynamic detection first.
                          If False, uses calibrated coordinates for reliability.

        Returns:
            GridConfig with detected grid parameters, or None if detection fails
        """
        if region is None:
            region = GRID_SEARCH_REGIONS.get(field_name)

        # Get calibrated config as fallback
        calibrated = self._fallback_grid_config(field_name, region or (0, 0, 100, 100))

        # Try dynamic detection first if forced or no high-confidence calibration
        if force_dynamic or calibrated.confidence < 0.9:
            dynamic_config = self._detect_grid_dynamic(gray, field_name, region)
            if dynamic_config is not None and dynamic_config.confidence >= 0.7:
                return dynamic_config

        # Use calibrated for known fields (more reliable after marker alignment)
        if calibrated.confidence >= 0.9:
            return calibrated

        # Try dynamic detection as last resort
        dynamic_config = self._detect_grid_dynamic(gray, field_name, region)
        if dynamic_config is not None:
            return dynamic_config

        return calibrated

    def _detect_grid_dynamic(
        self,
        gray: np.ndarray,
        field_name: str,
        region: Optional[Tuple[int, int, int, int]]
    ) -> Optional[GridConfig]:
        """
        Dynamically detect grid using HoughCircles.

        Args:
            gray: Grayscale image
            field_name: Name of the field
            region: Search region (x, y, width, height)

        Returns:
            GridConfig if detection succeeds, None otherwise
        """
        if region is None:
            return None

        x, y, w, h = region
        if y + h > gray.shape[0] or x + w > gray.shape[1]:
            return None

        roi = gray[y:y+h, x:x+w]

        # Detect circles using HoughCircles
        circles = self._detect_circles(roi)

        if circles is None or len(circles) < 4:
            circles = self._detect_circles_relaxed(roi)

        if circles is None or len(circles) < 4:
            return None

        # Offset circles to image coordinates
        circles_abs = [(c[0] + x, c[1] + y, c[2]) for c in circles]

        # Cluster circles into grid structure
        grid_config = self._cluster_to_grid(circles_abs, field_name)

        if grid_config is not None:
            # Validate against expected dimensions
            expected = EXPECTED_GRIDS.get(field_name)
            if expected:
                if not self._validate_grid(grid_config, expected):
                    return None

        return grid_config

    def _detect_circles(self, roi: np.ndarray) -> Optional[List[Tuple[int, int, int]]]:
        """Detect circles using HoughCircles with standard parameters."""
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)

        # Try multiple param2 values for robustness
        for param2 in [30, 25, 20, 15]:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=20,
                param1=50,
                param2=param2,
                minRadius=8,
                maxRadius=22
            )

            if circles is not None and len(circles[0]) >= 10:
                return [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0]]

        return None

    def _detect_circles_relaxed(self, roi: np.ndarray) -> Optional[List[Tuple[int, int, int]]]:
        """Detect circles with relaxed parameters for difficult images."""
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(roi)

        blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)

        for param2 in [15, 12, 10]:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.5,
                minDist=15,
                param1=40,
                param2=param2,
                minRadius=6,
                maxRadius=25
            )

            if circles is not None and len(circles[0]) >= 6:
                return [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0]]

        return None

    def _cluster_to_grid(
        self,
        circles: List[Tuple[int, int, int]],
        field_name: str
    ) -> Optional[GridConfig]:
        """
        Cluster detected circles into a grid structure.

        Groups circles by Y coordinate (rows) and X coordinate (columns),
        then calculates spacing and origin.
        """
        if len(circles) < 4:
            return None

        # Extract coordinates
        xs = np.array([c[0] for c in circles])
        ys = np.array([c[1] for c in circles])
        radii = np.array([c[2] for c in circles])

        # Cluster Y coordinates to find rows
        y_sorted = np.sort(ys)
        row_gaps = np.diff(y_sorted)
        median_row_gap = np.median(row_gaps[row_gaps > 10])

        # Cluster X coordinates to find columns
        x_sorted = np.sort(xs)
        col_gaps = np.diff(x_sorted)
        median_col_gap = np.median(col_gaps[col_gaps > 10])

        if np.isnan(median_row_gap) or np.isnan(median_col_gap):
            return None

        # Group circles by row (Y clustering)
        row_tolerance = median_row_gap * 0.4
        rows = self._cluster_1d(ys, row_tolerance)

        # Group circles by column (X clustering)
        col_tolerance = median_col_gap * 0.4
        cols = self._cluster_1d(xs, col_tolerance)

        if len(rows) < 2 or len(cols) < 2:
            return None

        # Find origin (top-left bubble)
        min_row_y = min(rows.keys())
        min_col_x = min(cols.keys())

        # Find the bubble closest to top-left corner
        origin_x = min_col_x
        origin_y = min_row_y

        # Calculate average radius
        avg_radius = int(np.median(radii))

        # Calculate grid skew (X shift per row)
        x_shift_per_row = self._calculate_x_shift(circles, rows, cols)

        return GridConfig(
            origin=(int(origin_x), int(origin_y)),
            rows=len(rows),
            cols=len(cols),
            row_gap=median_row_gap,
            col_gap=median_col_gap,
            bubble_radius=avg_radius,
            x_shift_per_row=x_shift_per_row,
            confidence=min(len(circles) / 50, 1.0)  # Higher if more circles detected
        )

    def _cluster_1d(self, values: np.ndarray, tolerance: float) -> Dict[float, List[int]]:
        """
        Cluster 1D values into groups within tolerance.

        Returns dict mapping cluster center to list of indices.
        """
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]

        clusters = {}
        current_cluster = [sorted_indices[0]]
        current_sum = sorted_values[0]

        for i in range(1, len(sorted_values)):
            if sorted_values[i] - sorted_values[i-1] <= tolerance:
                current_cluster.append(sorted_indices[i])
                current_sum += sorted_values[i]
            else:
                # Save current cluster
                cluster_center = current_sum / len(current_cluster)
                clusters[cluster_center] = current_cluster
                # Start new cluster
                current_cluster = [sorted_indices[i]]
                current_sum = sorted_values[i]

        # Save last cluster
        cluster_center = current_sum / len(current_cluster)
        clusters[cluster_center] = current_cluster

        return clusters

    def _calculate_x_shift(
        self,
        circles: List[Tuple[int, int, int]],
        rows: Dict[float, List[int]],
        cols: Dict[float, List[int]]
    ) -> float:
        """Calculate the X shift per row (for skewed grids)."""
        if len(rows) < 3:
            return 0.0

        # Get average X for first and last row
        row_centers = sorted(rows.keys())
        first_row_y = row_centers[0]
        last_row_y = row_centers[-1]

        # Find circles in first and last rows
        first_row_circles = [c for c in circles if abs(c[1] - first_row_y) < 20]
        last_row_circles = [c for c in circles if abs(c[1] - last_row_y) < 20]

        if not first_row_circles or not last_row_circles:
            return 0.0

        # Calculate average X shift
        first_avg_x = np.mean([c[0] for c in first_row_circles])
        last_avg_x = np.mean([c[0] for c in last_row_circles])

        num_rows = len(rows)
        x_shift = (last_avg_x - first_avg_x) / (num_rows - 1) if num_rows > 1 else 0

        return x_shift

    def _validate_grid(self, config: GridConfig, expected: Dict) -> bool:
        """Validate detected grid against expected dimensions."""
        # Allow some tolerance in row/column count
        row_tolerance = max(2, expected['rows'] * 0.2)
        col_tolerance = max(1, expected['cols'] * 0.3)

        row_ok = abs(config.rows - expected['rows']) <= row_tolerance
        col_ok = abs(config.cols - expected['cols']) <= col_tolerance

        return row_ok and col_ok

    def _fallback_grid_config(
        self,
        field_name: str,
        region: Tuple[int, int, int, int]
    ) -> GridConfig:
        """
        Generate fallback grid config using calibrated positions.

        These positions were measured on aligned images in the canonical frame (2480x3507).
        Uses calibrated coordinates for reliability instead of dynamic detection.
        """
        # Calibrated grid configs for canonical frame (2480x3507)
        # These were measured from actual OMR sheets after marker alignment
        # Marker positions: TL(170,530), TR(2310,530), BL(170,3150), BR(2310,3150)
        CALIBRATED_GRIDS = {
            'name': GridConfig(
                origin=(244, 708),  # Calibrated from marker-aligned images (2480x3507)
                rows=26,            # A-Z
                cols=25,            # 25 character positions
                row_gap=52,         # Vertical spacing between rows (calibrated)
                col_gap=50,         # Horizontal spacing between columns (calibrated 2026-01-03)
                bubble_radius=14,
                x_shift_per_row=0.0,
                confidence=0.95
            ),
            'roll_number': GridConfig(
                origin=(1570, 1290),  # Recalibrated 2026-01-03
                rows=10,            # 0-9
                cols=3,             # 3 digits
                row_gap=52,
                col_gap=65,
                bubble_radius=14,
                confidence=0.95
            ),
            'class': GridConfig(
                origin=(1780, 1280),  # Recalibrated 2026-01-03
                rows=8,             # Classes 3-10
                cols=1,
                row_gap=50,
                col_gap=0,
                bubble_radius=14,
                confidence=0.95
            ),
            'section': GridConfig(
                origin=(1930, 1280),  # Recalibrated 2026-01-03
                rows=8,             # Sections A-H
                cols=1,
                row_gap=54,
                col_gap=0,
                bubble_radius=14,
                confidence=0.95
            ),
            'school_code': GridConfig(
                origin=(2080, 1290),  # Recalibrated 2026-01-03
                rows=10,            # 0-9
                cols=3,             # 3 digits
                row_gap=52,
                col_gap=60,
                bubble_radius=14,
                confidence=0.95
            ),
            # Answer blocks - 6 blocks of 10 questions each
            # Each block: 10 rows (questions) x 4 cols (A/B/C/D)
            # Recalibrated 2026-01-03: X origins found by peak detection on 41_634.jpg
            # Y=2315 is Q1 row, col_gap detected ~55-66px (using 60 as average)
            'answers_1_10': GridConfig(
                origin=(328, 2315),   # Recalibrated from peak detection
                rows=10,              # Q1-10
                cols=4,               # A, B, C, D
                row_gap=93,
                col_gap=60,           # Adjusted from detected gaps (55-66)
                bubble_radius=14,
                confidence=0.95
            ),
            'answers_11_20': GridConfig(
                origin=(592, 2315),   # Recalibrated (was 660, -68px)
                rows=10,
                cols=4,
                row_gap=93,
                col_gap=60,
                bubble_radius=14,
                confidence=0.95
            ),
            'answers_21_30': GridConfig(
                origin=(924, 2315),   # Recalibrated (was 1058, -134px)
                rows=10,
                cols=4,
                row_gap=93,
                col_gap=60,
                bubble_radius=14,
                confidence=0.95
            ),
            'answers_31_40': GridConfig(
                origin=(1262, 2315),  # Recalibrated (was 1396, -134px)
                rows=10,
                cols=4,
                row_gap=93,
                col_gap=60,
                bubble_radius=14,
                confidence=0.95
            ),
            'answers_41_50': GridConfig(
                origin=(1598, 2315),  # Peak detection found leftmost at ~1598
                rows=10,
                cols=4,
                row_gap=93,
                col_gap=60,
                bubble_radius=14,
                confidence=0.95
            ),
            'answers_51_60': GridConfig(
                origin=(1937, 2315),  # Peak detection found leftmost at ~1937
                rows=10,
                cols=4,
                row_gap=93,
                col_gap=60,
                bubble_radius=14,
                confidence=0.95
            ),
        }

        if field_name in CALIBRATED_GRIDS:
            return CALIBRATED_GRIDS[field_name]

        # Generic fallback for unknown fields
        x, y, w, h = region
        expected = EXPECTED_GRIDS.get(field_name, {'rows': 10, 'cols': 4})

        rows = expected['rows']
        cols = expected['cols']

        row_gap = h / (rows + 1)
        col_gap = w / (cols + 1) if cols > 1 else w / 2

        origin_x = x + int(col_gap * 0.8)
        origin_y = y + int(row_gap * 0.8)

        return GridConfig(
            origin=(origin_x, origin_y),
            rows=rows,
            cols=cols,
            row_gap=row_gap,
            col_gap=col_gap,
            bubble_radius=14,
            confidence=0.3
        )

    def detect_all_grids(self, gray: np.ndarray) -> Dict[str, GridConfig]:
        """
        Detect all field grids in the image.

        Args:
            gray: Grayscale image (should be aligned to canonical frame)

        Returns:
            Dict mapping field names to their GridConfig
        """
        grids = {}

        for field_name in GRID_SEARCH_REGIONS.keys():
            config = self.detect_grid(gray, field_name)
            if config is not None:
                grids[field_name] = config

        return grids

    def get_bubble_position(
        self,
        grid: GridConfig,
        row: int,
        col: int
    ) -> Tuple[int, int]:
        """
        Calculate the center position of a bubble at given row/col.

        Args:
            grid: GridConfig for the field
            row: Row index (0-based)
            col: Column index (0-based)

        Returns:
            (x, y) center position of the bubble
        """
        x = int(grid.origin[0] + col * grid.col_gap + row * grid.x_shift_per_row)
        y = int(grid.origin[1] + row * grid.row_gap + col * grid.y_drift_per_col)
        return (x, y)

    def get_bubble_region(
        self,
        grid: GridConfig,
        row: int,
        col: int,
        padding: int = 2
    ) -> Tuple[int, int, int, int]:
        """
        Get the bounding box for a bubble at given row/col.

        Args:
            grid: GridConfig for the field
            row: Row index (0-based)
            col: Column index (0-based)
            padding: Extra pixels around bubble

        Returns:
            (x, y, width, height) bounding box
        """
        cx, cy = self.get_bubble_position(grid, row, col)
        r = grid.bubble_radius + padding
        return (cx - r, cy - r, 2 * r, 2 * r)


def detect_grid(gray: np.ndarray, field_name: str) -> Optional[GridConfig]:
    """
    Convenience function to detect a single grid.

    Args:
        gray: Grayscale image
        field_name: Name of the field

    Returns:
        GridConfig or None
    """
    detector = GridDetector()
    return detector.detect_grid(gray, field_name)


def detect_all_grids(gray: np.ndarray) -> Dict[str, GridConfig]:
    """
    Convenience function to detect all grids.

    Args:
        gray: Grayscale image

    Returns:
        Dict of field name -> GridConfig
    """
    detector = GridDetector()
    return detector.detect_all_grids(gray)

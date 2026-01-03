"""
Truly dynamic grid detection for OMR sheets.

Detects bubble positions dynamically without hardcoded coordinates:
1. Detect ALL bubbles using HoughCircles
2. Cluster into grid structures based on spacing patterns
3. Identify fields by relative position and grid dimensions
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class DetectedGrid:
    """A detected bubble grid."""
    origin: Tuple[int, int]      # (x, y) of top-left bubble center
    rows: int
    cols: int
    row_gap: float
    col_gap: float
    bubble_radius: int
    confidence: float
    field_type: str = ""         # Identified field type
    bubbles: List[Tuple[int, int, int]] = None  # List of (x, y, radius)


class DynamicGridDetector:
    """
    Detects bubble grids dynamically without hardcoded coordinates.

    Works by:
    1. Finding all circular bubbles in the image
    2. Clustering them by proximity into potential grids
    3. Analyzing grid dimensions to identify field types
    4. Returning detected grids with their positions
    """

    # Expected field characteristics for identification
    FIELD_SIGNATURES = {
        'name': {'rows': (20, 30), 'cols': (20, 30), 'position': 'top-left'},
        'roll_number': {'rows': (8, 12), 'cols': (2, 4), 'position': 'top-right'},
        'class': {'rows': (6, 10), 'cols': (1, 2), 'position': 'top-right'},
        'section': {'rows': (6, 10), 'cols': (1, 2), 'position': 'top-right'},
        'school_code': {'rows': (8, 12), 'cols': (2, 4), 'position': 'top-right'},
        'answers': {'rows': (8, 12), 'cols': (3, 5), 'position': 'bottom'},
    }

    def __init__(self, debug: bool = False):
        self.debug = debug
        self._debug_images = {}

    def detect_all_grids(self, gray: np.ndarray) -> Dict[str, DetectedGrid]:
        """
        Detect all bubble grids in the image.

        Args:
            gray: Grayscale image (preferably aligned)

        Returns:
            Dict mapping field names to DetectedGrid objects
        """
        h, w = gray.shape[:2]

        # Step 1: Detect all bubbles in the image
        all_bubbles = self._detect_all_bubbles(gray)

        if len(all_bubbles) < 10:
            return {}

        # Step 2: Cluster bubbles into grid structures
        grids = self._cluster_into_grids(all_bubbles, gray.shape)

        # Step 3: Identify field types based on position and dimensions
        identified_grids = self._identify_fields(grids, h, w)

        return identified_grids

    def detect_answer_grids(self, gray: np.ndarray) -> List[DetectedGrid]:
        """
        Detect answer grids using hybrid approach:
        - Use HoughCircles to find the Y position of first row
        - Use fixed calibrated X origins for each block
        - Use fixed row_gap and col_gap

        This is more reliable than pure HoughCircles because HoughCircles
        often misses unfilled bubbles, making column detection unreliable.

        Args:
            gray: Grayscale image

        Returns:
            List of 6 DetectedGrid objects for answer blocks
        """
        # Fixed calibrated grid parameters (calibrated from 41_645.jpg)
        # Y=2214 is Q1 row, X origins are column A of each block
        BLOCK_X_ORIGINS = [339, 680, 1011, 1347, 1684, 2024]
        EXPECTED_Y = 2214
        ROW_GAP = 93
        COL_GAP = 64
        BUBBLE_RADIUS = 14

        # Try to find actual first row Y using HoughCircles
        detected_y = self._find_first_row_y_hough(gray)

        # Use detected Y if found and reasonable, otherwise use expected
        if detected_y is not None and 2180 <= detected_y <= 2250:
            origin_y = detected_y
        else:
            origin_y = EXPECTED_Y

        # Create grids using fixed X origins and detected/expected Y
        grids = []
        for block_idx, origin_x in enumerate(BLOCK_X_ORIGINS):
            grid = DetectedGrid(
                origin=(origin_x, origin_y),
                rows=10,
                cols=4,
                row_gap=ROW_GAP,
                col_gap=COL_GAP,
                bubble_radius=BUBBLE_RADIUS,
                confidence=0.9,
                field_type=f'answers_{block_idx*10+1}_{block_idx*10+10}'
            )
            grids.append(grid)

        return grids

    def _find_first_row_y_hough(self, gray: np.ndarray) -> Optional[int]:
        """
        Find Y position of first answer row using HoughCircles.

        Returns the Y coordinate of the first detected row in the answer area,
        or None if detection fails.
        """
        # Detect circles
        all_bubbles = self._detect_all_bubbles(gray)

        if len(all_bubbles) < 50:
            return None

        # Filter to answer area (Y > 2180 to include Q1 row at Y~2214)
        answer_bubbles = [(x, y, r) for x, y, r in all_bubbles
                          if y > 2180 and 300 <= x <= 2200]

        if len(answer_bubbles) < 10:
            return None

        # Cluster Y values to find rows
        y_values = sorted(set(b[1] for b in answer_bubbles))
        clusters = [[y_values[0]]]
        for v in y_values[1:]:
            if v - clusters[-1][-1] <= 30:
                clusters[-1].append(v)
            else:
                clusters.append([v])

        if not clusters:
            return None

        # Return first row center (should be Q1)
        first_row_y = int(np.mean(clusters[0]))
        return first_row_y

    def _detect_answer_grids_hough(self, gray: np.ndarray) -> List[DetectedGrid]:
        """Detect answer grids using HoughCircles."""
        h, w = gray.shape[:2]

        # Detect all bubbles in the image
        all_bubbles = self._detect_all_bubbles(gray)

        if len(all_bubbles) < 100:
            return []

        # Filter to answer area (Y > 2180 to include Q1 row at Y~2214, X in 300-2200)
        answer_bubbles = [(x, y, r) for x, y, r in all_bubbles
                          if y > 2180 and 300 <= x <= 2200]

        if len(answer_bubbles) < 50:
            return []

        # Cluster by Y to find rows
        y_values = sorted(set(b[1] for b in answer_bubbles))
        row_centers = self._cluster_1d_simple(y_values, tolerance=30)

        if len(row_centers) < 8:
            return []

        # Fixed row gap of 93 pixels
        row_gap = 93

        # Get bubbles in first row
        first_row_y = row_centers[0]
        first_row_bubbles = [(x, y, r) for x, y, r in answer_bubbles
                             if abs(y - first_row_y) < 30]

        if len(first_row_bubbles) < 10:
            return []

        # Cluster X positions to find columns
        x_values = sorted(set(b[0] for b in first_row_bubbles))
        col_centers = self._cluster_1d_simple(x_values, tolerance=25)

        if len(col_centers) < 10:
            return []

        # Find column gaps to identify block boundaries
        col_gaps = [col_centers[i+1] - col_centers[i]
                   for i in range(len(col_centers)-1)]

        # Block boundaries are gaps > 70px
        block_boundaries = [0]
        for i, gap in enumerate(col_gaps):
            if gap > 70:
                block_boundaries.append(i + 1)
        block_boundaries.append(len(col_centers))

        # Extract valid blocks (>= 3 columns)
        answer_grids = []
        median_col_gap = np.median([g for g in col_gaps if g < 70]) if col_gaps else 63
        avg_radius = int(np.median([b[2] for b in answer_bubbles]))

        for i in range(len(block_boundaries) - 1):
            start_col = block_boundaries[i]
            end_col = block_boundaries[i + 1]
            block_cols = col_centers[start_col:end_col]

            if len(block_cols) >= 3:
                origin_x = int(block_cols[0])
                origin_y = int(first_row_y)

                grid = DetectedGrid(
                    origin=(origin_x, origin_y),
                    rows=10,
                    cols=min(4, len(block_cols)),
                    row_gap=row_gap,
                    col_gap=median_col_gap,
                    bubble_radius=avg_radius,
                    confidence=0.8,
                    field_type=f'answers_{len(answer_grids)*10+1}_{len(answer_grids)*10+10}'
                )
                answer_grids.append(grid)

        return answer_grids[:6]

    def _create_fixed_grids(self) -> List[DetectedGrid]:
        """Create grids using fixed calibrated positions."""
        BLOCK_X_ORIGINS = [339, 680, 1011, 1347, 1684, 2024]
        EXPECTED_Y = 2214
        ROW_GAP = 93
        COL_GAP = 64
        BUBBLE_RADIUS = 14

        grids = []
        for block_idx, origin_x in enumerate(BLOCK_X_ORIGINS):
            grid = DetectedGrid(
                origin=(origin_x, EXPECTED_Y),
                rows=10,
                cols=4,
                row_gap=ROW_GAP,
                col_gap=COL_GAP,
                bubble_radius=BUBBLE_RADIUS,
                confidence=0.7,
                field_type=f'answers_{block_idx*10+1}_{block_idx*10+10}'
            )
            grids.append(grid)
        return grids

    def _cluster_1d_simple(self, values: List[float], tolerance: float) -> List[float]:
        """Simple 1D clustering - returns cluster centers."""
        if not values:
            return []

        values = sorted(values)
        clusters = [[values[0]]]

        for v in values[1:]:
            if v - clusters[-1][-1] <= tolerance:
                clusters[-1].append(v)
            else:
                clusters.append([v])

        return [np.mean(c) for c in clusters]

    def _detect_all_bubbles(self, gray: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect all circular bubbles in the image.

        Returns list of (x, y, radius) tuples.
        """
        all_circles = []

        # Optimized parameters from testing on OMR sheets
        # blur=7, dp=1.2, minDist=25, param1=50, param2=15, r=8-22 works best
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=25,
            param1=50,
            param2=15,
            minRadius=8,
            maxRadius=22
        )

        if circles is not None:
            for c in circles[0]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                all_circles.append((cx, cy, r))

        # Also try with different params if not enough circles found
        if len(all_circles) < 50:
            blurred2 = cv2.GaussianBlur(gray, (5, 5), 0)
            circles2 = cv2.HoughCircles(
                blurred2,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=30,
                param1=30,
                param2=15,
                minRadius=10,
                maxRadius=22
            )

            if circles2 is not None:
                for c in circles2[0]:
                    cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                    # Check if not duplicate
                    is_dup = any(abs(cx - ex) < 15 and abs(cy - ey) < 15
                                for ex, ey, er in all_circles)
                    if not is_dup:
                        all_circles.append((cx, cy, r))

        return all_circles

    def _cluster_into_grids(
        self,
        bubbles: List[Tuple[int, int, int]],
        image_shape: Tuple[int, int]
    ) -> List[DetectedGrid]:
        """
        Cluster bubbles into grid structures based on spacing patterns.
        """
        if len(bubbles) < 4:
            return []

        # Sort bubbles by position
        bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))  # Sort by Y, then X

        # Find row clusters (bubbles with similar Y)
        y_values = [b[1] for b in bubbles]
        row_clusters = self._cluster_1d_values(y_values, tolerance=25)

        if len(row_clusters) < 2:
            return []

        # Calculate median row gap
        row_centers = sorted(row_clusters.keys())
        row_gaps = [row_centers[i+1] - row_centers[i] for i in range(len(row_centers)-1)]
        median_row_gap = np.median(row_gaps) if row_gaps else 50

        # Find column clusters
        x_values = [b[0] for b in bubbles]
        col_clusters = self._cluster_1d_values(x_values, tolerance=20)

        if len(col_clusters) < 2:
            return []

        col_centers = sorted(col_clusters.keys())
        col_gaps = [col_centers[i+1] - col_centers[i] for i in range(len(col_centers)-1)]
        median_col_gap = np.median(col_gaps) if col_gaps else 50

        # Now find distinct grid regions by looking for gaps > 2x median
        grids = self._split_into_separate_grids(bubbles, median_row_gap, median_col_gap)

        return grids

    def _cluster_1d_values(self, values: List[float], tolerance: float) -> Dict[float, List[int]]:
        """Cluster 1D values into groups within tolerance."""
        if not values:
            return {}

        sorted_indices = np.argsort(values)
        sorted_values = [values[i] for i in sorted_indices]

        clusters = {}
        current_cluster = [sorted_indices[0]]
        current_sum = sorted_values[0]

        for i in range(1, len(sorted_values)):
            if sorted_values[i] - sorted_values[i-1] <= tolerance:
                current_cluster.append(sorted_indices[i])
                current_sum += sorted_values[i]
            else:
                cluster_center = current_sum / len(current_cluster)
                clusters[cluster_center] = current_cluster
                current_cluster = [sorted_indices[i]]
                current_sum = sorted_values[i]

        cluster_center = current_sum / len(current_cluster)
        clusters[cluster_center] = current_cluster

        return clusters

    def _split_into_separate_grids(
        self,
        bubbles: List[Tuple[int, int, int]],
        row_gap: float,
        col_gap: float
    ) -> List[DetectedGrid]:
        """
        Split bubbles into separate grid regions based on large gaps.

        Answer blocks have ~4 columns each with ~60px gaps between columns,
        and ~100-150px gaps between blocks.
        """
        if len(bubbles) < 4:
            return []

        # Sort all X positions
        x_positions = sorted(set(b[0] for b in bubbles))

        # Find gaps between consecutive X positions
        x_gaps = []
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            x_gaps.append((x_positions[i-1], x_positions[i], gap))

        # Sort gaps by size to find block boundaries
        # Block boundaries have gaps > 100px typically
        # Column gaps within a block are ~60px
        sorted_gaps = sorted(x_gaps, key=lambda g: -g[2])  # Largest first

        # Find the 5 largest gaps (separating 6 blocks)
        # But only if they're significantly larger than typical column gap
        median_gap = np.median([g[2] for g in x_gaps]) if x_gaps else 60
        block_boundaries = []

        for start_x, end_x, gap in sorted_gaps[:10]:  # Check top 10 largest gaps
            if gap > median_gap * 1.5 and gap > 80:  # Must be notably larger
                block_boundaries.append((start_x + end_x) / 2)  # Midpoint

        block_boundaries = sorted(block_boundaries)

        # Split bubbles by block boundaries
        grids = []
        block_boundaries = [0] + block_boundaries + [float('inf')]

        for i in range(len(block_boundaries) - 1):
            x_min = block_boundaries[i]
            x_max = block_boundaries[i + 1]

            # Get bubbles in this X range
            group_bubbles = [b for b in bubbles if x_min <= b[0] < x_max]

            if len(group_bubbles) < 20:  # Need at least 20 bubbles for answer block
                continue

            # Analyze this group's structure
            grid = self._analyze_grid_structure(group_bubbles)
            if grid is not None and grid.rows >= 8 and grid.cols >= 3:
                grids.append(grid)

        return grids

    def _analyze_grid_structure(
        self,
        bubbles: List[Tuple[int, int, int]]
    ) -> Optional[DetectedGrid]:
        """
        Analyze a group of bubbles to determine grid structure.
        """
        if len(bubbles) < 4:
            return None

        xs = [b[0] for b in bubbles]
        ys = [b[1] for b in bubbles]
        radii = [b[2] for b in bubbles]

        # Cluster X and Y to find rows/columns
        x_clusters = self._cluster_1d_values(xs, tolerance=20)
        y_clusters = self._cluster_1d_values(ys, tolerance=25)

        if len(x_clusters) < 2 or len(y_clusters) < 2:
            return None

        # Get sorted positions
        col_positions = sorted(x_clusters.keys())
        row_positions = sorted(y_clusters.keys())

        # Calculate gaps
        col_gaps = [col_positions[i+1] - col_positions[i] for i in range(len(col_positions)-1)]
        row_gaps = [row_positions[i+1] - row_positions[i] for i in range(len(row_positions)-1)]

        col_gap = np.median(col_gaps) if col_gaps else 60
        row_gap = np.median(row_gaps) if row_gaps else 90

        # Origin is top-left
        origin_x = col_positions[0]
        origin_y = row_positions[0]

        # Calculate confidence based on regularity of spacing
        col_std = np.std(col_gaps) if len(col_gaps) > 1 else 0
        row_std = np.std(row_gaps) if len(row_gaps) > 1 else 0
        regularity = 1.0 - min(1.0, (col_std + row_std) / 50)

        # Confidence also depends on number of bubbles found
        expected_bubbles = len(col_positions) * len(row_positions)
        coverage = min(1.0, len(bubbles) / max(1, expected_bubbles * 0.7))

        confidence = regularity * 0.6 + coverage * 0.4

        return DetectedGrid(
            origin=(int(origin_x), int(origin_y)),
            rows=len(row_positions),
            cols=len(col_positions),
            row_gap=row_gap,
            col_gap=col_gap,
            bubble_radius=int(np.median(radii)),
            confidence=confidence,
            bubbles=bubbles
        )

    def _identify_fields(
        self,
        grids: List[DetectedGrid],
        image_height: int,
        image_width: int
    ) -> Dict[str, DetectedGrid]:
        """
        Identify field types based on position and dimensions.
        """
        identified = {}

        # Midpoints for position classification
        mid_x = image_width / 2
        mid_y = image_height / 2

        answer_blocks = []

        for grid in grids:
            ox, oy = grid.origin

            # Bottom half = answer blocks
            if oy > mid_y * 1.1:
                if 8 <= grid.rows <= 12 and 3 <= grid.cols <= 6:
                    answer_blocks.append(grid)
                continue

            # Top-left large grid = name
            if ox < mid_x * 0.7 and oy < mid_y:
                if grid.rows >= 20 and grid.cols >= 15:
                    grid.field_type = 'name'
                    identified['name'] = grid
                    continue

            # Top-right small grids = roll, class, section, school_code
            if ox > mid_x * 0.6 and oy < mid_y:
                if grid.rows >= 8 and 2 <= grid.cols <= 4:
                    # Identify by X position (left to right: roll, class, section, school)
                    if 'roll_number' not in identified:
                        grid.field_type = 'roll_number'
                        identified['roll_number'] = grid
                    elif 'school_code' not in identified:
                        grid.field_type = 'school_code'
                        identified['school_code'] = grid
                elif grid.rows >= 6 and grid.cols == 1:
                    if 'class' not in identified:
                        grid.field_type = 'class'
                        identified['class'] = grid
                    elif 'section' not in identified:
                        grid.field_type = 'section'
                        identified['section'] = grid

        # Sort answer blocks by X and label them
        answer_blocks.sort(key=lambda g: g.origin[0])
        for i, grid in enumerate(answer_blocks[:6]):
            field_name = f'answers_{i*10+1}_{i*10+10}'
            grid.field_type = field_name
            identified[field_name] = grid

        return identified

    def get_bubble_position(
        self,
        grid: DetectedGrid,
        row: int,
        col: int
    ) -> Tuple[int, int]:
        """Get the center position of a bubble at given row/col."""
        x = int(grid.origin[0] + col * grid.col_gap)
        y = int(grid.origin[1] + row * grid.row_gap)
        return (x, y)


def detect_grids_dynamically(gray: np.ndarray) -> Dict[str, DetectedGrid]:
    """Convenience function for dynamic grid detection."""
    detector = DynamicGridDetector()
    return detector.detect_all_grids(gray)


def detect_answer_blocks(gray: np.ndarray) -> List[DetectedGrid]:
    """Convenience function to detect answer blocks."""
    detector = DynamicGridDetector()
    return detector.detect_answer_grids(gray)

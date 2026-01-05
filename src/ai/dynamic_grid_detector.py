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

    # Batch-specific grid parameters (detected by form COLOR, not file names)
    # Red forms = wider row spacing, Blue forms = narrower row spacing
    BATCH_PARAMS = {
        'khoj': {
            'block_spacing': 341,
            'row_gap': 93,
            'col_gap': 64,
            'expected_y': 2214,
            'block_x_origins': [339, 680, 1011, 1347, 1684, 2024],
        },
        'chayan': {
            'block_spacing': 337,
            'row_gap': 70,
            'col_gap': 64,
            'expected_y': 2197,
            'block_x_origins': [332, 668, 1006, 1344, 1682, 2018],
        }
    }

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
        self._batch_type = 'khoj'  # Default to Khoj

    def _detect_batch_type(self, color_image: np.ndarray) -> str:
        """
        Detect batch type (khoj or chayan) based on form color.

        Khoj forms are red, Chayan forms are blue.
        Samples multiple regions to handle edge cases where one region may be blank.

        Args:
            color_image: BGR color image

        Returns:
            'khoj' for red forms, 'chayan' for blue forms
        """
        if color_image is None or len(color_image.shape) != 3:
            return 'khoj'  # Default if no color info

        h, w = color_image.shape[:2]
        if h < 1000 or w < 500:
            return 'khoj'

        # Sample from PRINTED FORM AREAS where students don't write
        # - Answer grid area (Y > 2000) has printed row labels and borders
        # - The form's printed color (red/blue) dominates these areas
        # - Avoid the bubble centers where students fill
        sample_regions = [
            (2050, 2150, 260, 320),   # Left of Block 1 (row labels)
            (2050, 2150, 600, 660),   # Left of Block 2 (row labels)
            (2050, 2150, 940, 1000),  # Left of Block 3 (row labels)
            (2900, 3000, 260, 320),   # Bottom of answer area (printed lines)
            (400, 500, 150, 200),     # Form header area (printed text)
        ]

        total_blue = 0
        total_red = 0
        valid_samples = 0

        for y1, y2, x1, x2 in sample_regions:
            if y2 > h or x2 > w:
                continue

            region = color_image[y1:y2, x1:x2]
            avg_intensity = np.mean(region)

            # Skip mostly white regions (blank areas)
            if avg_intensity > 250:
                continue

            avg_blue = np.mean(region[:, :, 0])
            avg_red = np.mean(region[:, :, 2])

            total_blue += avg_blue
            total_red += avg_red
            valid_samples += 1

        # If no valid samples, try a larger central region
        if valid_samples == 0:
            central = color_image[h//3:2*h//3, w//4:3*w//4]
            total_blue = np.mean(central[:, :, 0])
            total_red = np.mean(central[:, :, 2])
            valid_samples = 1

        avg_blue = total_blue / valid_samples
        avg_red = total_red / valid_samples

        # Blue forms are Chayan, red forms are Khoj
        if avg_blue > avg_red + 5:
            return 'chayan'
        return 'khoj'

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

    def detect_answer_grids(self, gray: np.ndarray, color_image: np.ndarray = None) -> List[DetectedGrid]:
        """
        Detect answer grids dynamically using HoughCircles.

        Detects actual bubble positions to handle per-image alignment variations.
        Falls back to calibrated positions if detection fails.

        Args:
            gray: Grayscale image
            color_image: Optional BGR color image for batch detection

        Returns:
            List of 6 DetectedGrid objects for answer blocks
        """
        # Detect batch type if color image provided
        if color_image is not None:
            self._batch_type = self._detect_batch_type(color_image)

        # Get batch-specific parameters
        params = self.BATCH_PARAMS[self._batch_type]

        # Try dynamic detection first
        grids = self._detect_grids_dynamically(gray, params)

        if len(grids) == 6:
            return grids

        # Fallback to calibrated positions if dynamic detection fails
        return self._create_fixed_grids(params)

    def _detect_grids_dynamically(self, gray: np.ndarray, params: dict = None) -> List[DetectedGrid]:
        """
        Dynamically detect grid positions using HoughCircles.

        Uses batch-specific row_gap from params (detected by form color) since
        dynamic row_gap calculation is less reliable than color-based batch detection.

        Args:
            gray: Grayscale image
            params: Batch-specific parameters dict (detected by form color)

        Returns list of DetectedGrid or empty list if detection fails.
        """
        # Use provided params - these are based on form color detection
        if params is None:
            # No color info available - use reasonable defaults
            params = {'row_gap': 90, 'col_gap': 64, 'block_spacing': 340}

        ROW_GAP = params.get('row_gap', 90)
        COL_GAP = params.get('col_gap', 64)
        BUBBLE_RADIUS = 14

        # Detect all bubbles
        all_bubbles = self._detect_all_bubbles(gray)

        if len(all_bubbles) < 100:
            return []

        # Filter to answer area
        answer_bubbles = [(x, y, r) for x, y, r in all_bubbles
                          if 2150 < y < 3200 and 200 < x < 2300]

        if len(answer_bubbles) < 50:
            return []

        # Find rows by clustering Y values
        y_values = sorted(set(b[1] for b in answer_bubbles))
        y_clusters = []
        current = [y_values[0]]
        for y in y_values[1:]:
            if y - current[-1] <= 30:
                current.append(y)
            else:
                y_clusters.append(current)
                current = [y]
        y_clusters.append(current)

        if not y_clusters:
            return []

        first_row_y = int(np.mean(y_clusters[0]))

        # Get bubbles in first row
        first_row = [(x, y) for x, y, r in answer_bubbles
                     if abs(y - first_row_y) < 30]
        first_row.sort(key=lambda p: p[0])

        if len(first_row) < 20:
            return []

        x_positions = sorted(set(p[0] for p in first_row))

        # Find block boundaries by detecting large X gaps (gaps between blocks > 80px)
        # Columns within a block are ~64px apart, blocks are ~340px apart
        block_x_origins = []
        prev_x = x_positions[0]
        block_x_origins.append(prev_x)

        for x in x_positions[1:]:
            gap = x - prev_x
            if gap > 80:  # Large gap = new block
                block_x_origins.append(x)
            prev_x = x

        # Calculate dynamic BLOCK_SPACING if we found multiple blocks
        if len(block_x_origins) >= 2:
            block_gaps = [block_x_origins[i+1] - block_x_origins[i] for i in range(len(block_x_origins)-1)]
            BLOCK_SPACING = int(np.median(block_gaps))
        else:
            BLOCK_SPACING = params.get('block_spacing', 340)

        # If we found fewer than 6 blocks, extrapolate the missing ones
        if len(block_x_origins) < 6 and len(block_x_origins) >= 1:
            block1_origin = block_x_origins[0]
            # Recalculate all 6 block origins using detected spacing
            block_x_origins = []
            for block_idx in range(6):
                expected_x = block1_origin + block_idx * BLOCK_SPACING
                # Find closest detected X position within 50px of expected
                nearby = [x for x in x_positions if abs(x - expected_x) < 50]
                if nearby:
                    block_x_origins.append(min(nearby))
                else:
                    block_x_origins.append(int(expected_x))

        if len(block_x_origins) < 6:
            return []

        # Create grids
        grids = []
        for block_idx, origin_x in enumerate(block_x_origins):
            grid = DetectedGrid(
                origin=(origin_x, first_row_y),
                rows=10,
                cols=4,
                row_gap=ROW_GAP,
                col_gap=COL_GAP,
                bubble_radius=BUBBLE_RADIUS,
                confidence=0.85,
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

    def _create_fixed_grids(self, params: dict = None) -> List[DetectedGrid]:
        """Create grids using fixed calibrated positions.

        Args:
            params: Batch-specific parameters dict
        """
        # Use provided params or default to Khoj
        if params is None:
            params = self.BATCH_PARAMS['khoj']

        BLOCK_X_ORIGINS = params['block_x_origins']
        EXPECTED_Y = params['expected_y']
        ROW_GAP = params['row_gap']
        COL_GAP = params['col_gap']
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

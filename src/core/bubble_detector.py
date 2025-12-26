"""
Bubble detection algorithms
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from src.config.constants import DetectionThresholds, BubbleStatus
from src.utils.image_utils import ImageUtils
from src.utils.logging_utils import get_logger

logger = get_logger('bubble_detector')


@dataclass
class BubbleResult:
    """Result of analyzing a single bubble"""
    x: int
    y: int
    width: int
    height: int
    value: str
    label: str
    mean_intensity: float
    fill_ratio: float
    status: BubbleStatus
    is_filled: bool


class BubbleDetector:
    """Detects and analyzes bubble marks on OMR sheets"""

    def __init__(self,
                 fill_threshold: int = DetectionThresholds.FILL_THRESHOLD,
                 min_fill_ratio: float = DetectionThresholds.MIN_FILL_RATIO,
                 partial_threshold: float = DetectionThresholds.PARTIAL_THRESHOLD):
        """
        Initialize bubble detector

        Args:
            fill_threshold: Mean intensity threshold (0-255, lower = darker = filled)
            min_fill_ratio: Minimum ratio of dark pixels for filled bubble
            partial_threshold: Threshold for flagging partial fills
        """
        self.fill_threshold = fill_threshold
        self.min_fill_ratio = min_fill_ratio
        self.partial_threshold = partial_threshold

    def analyze_bubble(self,
                       image: np.ndarray,
                       x: int, y: int,
                       width: int, height: int,
                       value: str = "",
                       label: str = "") -> BubbleResult:
        """
        Analyze a single bubble region

        Args:
            image: Grayscale image
            x, y: Top-left corner coordinates
            width, height: Bubble dimensions
            value: Bubble value (e.g., 'A', 'B', 'C', 'D')
            label: Field label (e.g., 'q1', 'Name_1')

        Returns:
            BubbleResult with analysis data
        """
        # Extract ROI
        roi = ImageUtils.extract_roi(image, x, y, width, height)

        if roi.size == 0:
            logger.warning(f"Empty ROI at ({x}, {y})")
            return BubbleResult(
                x=x, y=y, width=width, height=height,
                value=value, label=label,
                mean_intensity=255, fill_ratio=0,
                status=BubbleStatus.EMPTY, is_filled=False
            )

        # Calculate metrics
        mean_intensity = ImageUtils.calculate_mean_intensity(roi)
        fill_ratio = ImageUtils.calculate_fill_ratio(roi, self.fill_threshold)

        # Determine status
        if fill_ratio >= self.min_fill_ratio:
            status = BubbleStatus.FILLED
            is_filled = True
        elif fill_ratio >= self.partial_threshold:
            status = BubbleStatus.PARTIAL
            is_filled = False  # Do NOT count partial as filled to avoid false positives
        else:
            status = BubbleStatus.EMPTY
            is_filled = False

        return BubbleResult(
            x=x, y=y, width=width, height=height,
            value=value, label=label,
            mean_intensity=mean_intensity,
            fill_ratio=fill_ratio,
            status=status,
            is_filled=is_filled
        )

    def analyze_field_bubbles(self,
                               image: np.ndarray,
                               bubble_coords: List[Dict[str, Any]]) -> List[BubbleResult]:
        """
        Analyze all bubbles in a field

        Args:
            image: Grayscale image
            bubble_coords: List of bubble coordinate dicts

        Returns:
            List of BubbleResult for each bubble
        """
        results = []

        for coord in bubble_coords:
            result = self.analyze_bubble(
                image,
                x=coord['x'],
                y=coord['y'],
                width=coord['width'],
                height=coord['height'],
                value=coord['value'],
                label=coord['label']
            )
            results.append(result)

        return results

    def detect_marked_value(self,
                             image: np.ndarray,
                             bubble_coords: List[Dict[str, Any]],
                             group_by_label: bool = True) -> Dict[str, str]:
        """
        Detect marked values for a field block

        Args:
            image: Grayscale image
            bubble_coords: List of bubble coordinates
            group_by_label: Group results by label

        Returns:
            Dict mapping labels to detected values
        """
        # Analyze all bubbles
        bubble_results = self.analyze_field_bubbles(image, bubble_coords)

        # Group by label
        labels_to_bubbles: Dict[str, List[BubbleResult]] = {}
        for result in bubble_results:
            if result.label not in labels_to_bubbles:
                labels_to_bubbles[result.label] = []
            labels_to_bubbles[result.label].append(result)

        # Determine value for each label
        detected_values = {}

        for label, bubbles in labels_to_bubbles.items():
            filled_bubbles = [b for b in bubbles if b.is_filled]

            if len(filled_bubbles) == 0:
                # No bubble filled
                detected_values[label] = ""
            elif len(filled_bubbles) == 1:
                # Exactly one filled
                detected_values[label] = filled_bubbles[0].value
            else:
                # Multiple filled - use the one with highest fill ratio
                # but flag as multi-mark
                best = max(filled_bubbles, key=lambda b: b.fill_ratio)

                # Check if it's truly multi-marked (more than one significantly filled)
                significant_fills = [b for b in filled_bubbles
                                     if b.fill_ratio >= self.min_fill_ratio]

                if len(significant_fills) > 1:
                    # Sort by value and take first (A before B, 0 before 1)
                    sorted_fills = sorted(significant_fills, key=lambda b: b.value)
                    detected_values[label] = sorted_fills[0].value
                else:
                    detected_values[label] = best.value

        return detected_values

    def detect_best_match(self,
                          image: np.ndarray,
                          bubble_coords: List[Dict[str, Any]],
                          label: str) -> Tuple[str, float]:
        """
        Detect best matching value for a single field

        Args:
            image: Grayscale image
            bubble_coords: Bubble coordinates for this field
            label: Field label to check

        Returns:
            Tuple of (detected_value, confidence)
        """
        # Filter bubbles for this label
        label_bubbles = [c for c in bubble_coords if c['label'] == label]

        if not label_bubbles:
            return ("", 0.0)

        # Analyze each bubble
        results = []
        for coord in label_bubbles:
            result = self.analyze_bubble(
                image,
                x=coord['x'],
                y=coord['y'],
                width=coord['width'],
                height=coord['height'],
                value=coord['value'],
                label=coord['label']
            )
            results.append(result)

        # Find best match (lowest mean intensity or highest fill ratio)
        filled_results = [r for r in results if r.is_filled]

        if not filled_results:
            return ("", 0.0)

        # Sort by fill ratio (highest first)
        filled_results.sort(key=lambda r: r.fill_ratio, reverse=True)

        best = filled_results[0]

        # Check for multi-mark
        if len(filled_results) > 1:
            second_best = filled_results[1]
            # If second best is also significantly filled, it's multi-mark
            if second_best.fill_ratio >= self.min_fill_ratio * 0.8:
                return ("MULTI", best.fill_ratio)

        return (best.value, best.fill_ratio)

    def calibrate_threshold(self,
                            image: np.ndarray,
                            known_filled_coords: List[Dict[str, Any]],
                            known_empty_coords: List[Dict[str, Any]]) -> int:
        """
        Calibrate fill threshold based on known filled and empty bubbles

        Args:
            image: Grayscale image
            known_filled_coords: Coordinates of bubbles known to be filled
            known_empty_coords: Coordinates of bubbles known to be empty

        Returns:
            Optimal threshold value
        """
        filled_intensities = []
        empty_intensities = []

        for coord in known_filled_coords:
            roi = ImageUtils.extract_roi(
                image, coord['x'], coord['y'],
                coord['width'], coord['height']
            )
            if roi.size > 0:
                filled_intensities.append(ImageUtils.calculate_mean_intensity(roi))

        for coord in known_empty_coords:
            roi = ImageUtils.extract_roi(
                image, coord['x'], coord['y'],
                coord['width'], coord['height']
            )
            if roi.size > 0:
                empty_intensities.append(ImageUtils.calculate_mean_intensity(roi))

        if filled_intensities and empty_intensities:
            # Optimal threshold is midpoint between filled and empty means
            filled_mean = np.mean(filled_intensities)
            empty_mean = np.mean(empty_intensities)
            optimal_threshold = int((filled_mean + empty_mean) / 2)

            logger.info(f"Calibrated threshold: {optimal_threshold} "
                       f"(filled mean: {filled_mean:.1f}, empty mean: {empty_mean:.1f})")

            return optimal_threshold

        return self.fill_threshold

    def set_threshold(self, threshold: int) -> None:
        """Update fill threshold"""
        self.fill_threshold = threshold
        logger.info(f"Fill threshold set to: {threshold}")

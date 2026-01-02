"""
AI-based bubble detection using YOLO.

Detects filled and empty bubbles without relying on fixed coordinates.
Uses object detection to find bubbles anywhere on the sheet.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class DetectedBubble:
    """A detected bubble on the OMR sheet."""
    x: int
    y: int
    width: int
    height: int
    is_filled: bool
    confidence: float
    row_index: int = -1  # Will be assigned after grid analysis
    col_index: int = -1


@dataclass
class BubbleGrid:
    """A grid of bubbles for a specific field (e.g., name, roll number)."""
    name: str
    bubbles: List[List[DetectedBubble]]  # [row][col]
    origin: Tuple[int, int]
    rows: int
    cols: int


class AIBubbleDetector:
    """
    AI-powered bubble detector using deep learning.

    Can use either:
    1. YOLO model for bubble detection (requires training)
    2. Traditional CV with learned thresholds (faster, no training)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = False
    ):
        """
        Initialize the bubble detector.

        Args:
            model_path: Path to ONNX model file (if using trained model)
            use_gpu: Whether to use GPU acceleration
        """
        self.model = None
        self.use_gpu = use_gpu

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

        # Default thresholds (learned from training data)
        self.fill_threshold = 0.4  # Ratio of dark pixels for filled bubble
        self.bubble_size_range = (20, 50)  # Min/max bubble diameter

    def _load_model(self, model_path: str):
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            self.model = ort.InferenceSession(model_path, providers=providers)
        except ImportError:
            print("ONNX Runtime not available, using traditional CV")
            self.model = None

    def detect_bubbles(
        self,
        image: np.ndarray,
        region: Optional[Tuple[int, int, int, int]] = None
    ) -> List[DetectedBubble]:
        """
        Detect all bubbles in the image or specified region.

        Args:
            image: Input BGR image (aligned)
            region: Optional (x, y, width, height) to limit detection

        Returns:
            List of detected bubbles
        """
        if region:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            offset_x, offset_y = x, y
        else:
            roi = image
            offset_x, offset_y = 0, 0

        if self.model:
            return self._detect_with_model(roi, offset_x, offset_y)
        else:
            return self._detect_traditional(roi, offset_x, offset_y)

    def _detect_with_model(
        self,
        image: np.ndarray,
        offset_x: int,
        offset_y: int
    ) -> List[DetectedBubble]:
        """Detect bubbles using YOLO model."""
        # Preprocess image for YOLO
        input_size = 640  # Standard YOLO input size
        h, w = image.shape[:2]
        scale = input_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Pad to square
        padded = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Normalize and transpose for ONNX
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(2, 0, 1)
        input_tensor = np.expand_dims(input_tensor, 0)

        # Run inference
        outputs = self.model.run(None, {'images': input_tensor})

        # Parse YOLO output
        bubbles = []
        detections = outputs[0][0]  # Shape: [num_detections, 6] (x, y, w, h, conf, class)

        for det in detections:
            x, y, bw, bh, conf, cls = det
            if conf > 0.5:  # Confidence threshold
                # Convert back to original coordinates
                bx = int((x - bw/2) / scale) + offset_x
                by = int((y - bh/2) / scale) + offset_y
                bw = int(bw / scale)
                bh = int(bh / scale)

                is_filled = int(cls) == 1  # Class 0 = empty, 1 = filled

                bubbles.append(DetectedBubble(
                    x=bx, y=by, width=bw, height=bh,
                    is_filled=is_filled, confidence=float(conf)
                ))

        return bubbles

    def _detect_traditional(
        self,
        image: np.ndarray,
        offset_x: int,
        offset_y: int
    ) -> List[DetectedBubble]:
        """
        Detect bubbles using traditional computer vision.

        Uses circle detection and fill analysis.
        """
        bubbles = []

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.bubble_size_range[0],
            param1=50,
            param2=30,
            minRadius=self.bubble_size_range[0] // 2,
            maxRadius=self.bubble_size_range[1] // 2
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0]:
                cx, cy, r = circle

                # Extract bubble region
                x1 = max(0, cx - r)
                y1 = max(0, cy - r)
                x2 = min(gray.shape[1], cx + r)
                y2 = min(gray.shape[0], cy + r)

                bubble_region = gray[y1:y2, x1:x2]

                if bubble_region.size == 0:
                    continue

                # Analyze fill
                is_filled, confidence = self._analyze_bubble_fill(bubble_region)

                bubbles.append(DetectedBubble(
                    x=int(x1 + offset_x),
                    y=int(y1 + offset_y),
                    width=int(x2 - x1),
                    height=int(y2 - y1),
                    is_filled=is_filled,
                    confidence=confidence
                ))

        return bubbles

    def _analyze_bubble_fill(self, bubble_region: np.ndarray) -> Tuple[bool, float]:
        """
        Analyze if a bubble is filled.

        Returns:
            Tuple of (is_filled, confidence)
        """
        # Apply threshold
        _, binary = cv2.threshold(bubble_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Calculate fill ratio
        total_pixels = binary.size
        dark_pixels = np.sum(binary > 0)
        fill_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0

        # Determine if filled
        is_filled = fill_ratio > self.fill_threshold

        # Confidence based on how clearly filled/empty
        if is_filled:
            confidence = min(1.0, fill_ratio / self.fill_threshold)
        else:
            confidence = min(1.0, (self.fill_threshold - fill_ratio) / self.fill_threshold)

        return is_filled, confidence

    def detect_grid(
        self,
        image: np.ndarray,
        grid_config: Dict
    ) -> BubbleGrid:
        """
        Detect bubbles in a grid pattern (for name, roll number, etc.).

        Args:
            image: Aligned input image
            grid_config: Configuration with expected rows, cols, and region

        Returns:
            BubbleGrid with organized bubbles
        """
        region = grid_config.get('region')
        expected_rows = grid_config.get('rows', 26)
        expected_cols = grid_config.get('cols', 25)
        name = grid_config.get('name', 'unknown')

        # Detect all bubbles in region
        bubbles = self.detect_bubbles(image, region)

        if not bubbles:
            return BubbleGrid(name=name, bubbles=[], origin=(0, 0), rows=0, cols=0)

        # Cluster bubbles into rows and columns
        organized = self._organize_into_grid(bubbles, expected_rows, expected_cols)

        # Determine grid origin
        if organized and organized[0]:
            origin = (organized[0][0].x, organized[0][0].y)
        else:
            origin = (0, 0)

        return BubbleGrid(
            name=name,
            bubbles=organized,
            origin=origin,
            rows=len(organized),
            cols=len(organized[0]) if organized else 0
        )

    def _organize_into_grid(
        self,
        bubbles: List[DetectedBubble],
        expected_rows: int,
        expected_cols: int
    ) -> List[List[DetectedBubble]]:
        """Organize detected bubbles into a 2D grid."""
        if not bubbles:
            return []

        # Sort bubbles by y-coordinate first (rows)
        bubbles_sorted = sorted(bubbles, key=lambda b: b.y)

        # Cluster into rows
        rows = []
        current_row = [bubbles_sorted[0]]
        row_threshold = bubbles_sorted[0].height * 0.5

        for bubble in bubbles_sorted[1:]:
            if abs(bubble.y - current_row[0].y) < row_threshold:
                current_row.append(bubble)
            else:
                # Sort current row by x and add
                current_row.sort(key=lambda b: b.x)
                rows.append(current_row)
                current_row = [bubble]

        # Add last row
        current_row.sort(key=lambda b: b.x)
        rows.append(current_row)

        # Assign row and column indices
        for row_idx, row in enumerate(rows):
            for col_idx, bubble in enumerate(row):
                bubble.row_index = row_idx
                bubble.col_index = col_idx

        return rows

    def extract_field_value(
        self,
        grid: BubbleGrid,
        values: List[str]
    ) -> str:
        """
        Extract the value from a bubble grid.

        For single-column fields (like roll number digit), returns the filled value.
        For multi-column fields (like name), returns concatenated values.

        Args:
            grid: Detected bubble grid
            values: Possible values for each row (e.g., ['A', 'B', ..., 'Z'])

        Returns:
            Extracted string value
        """
        result = []

        # For each column, find the filled bubble
        if not grid.bubbles:
            return ""

        # Transpose if needed to iterate by columns
        max_cols = max(len(row) for row in grid.bubbles) if grid.bubbles else 0

        for col_idx in range(max_cols):
            filled_row = None
            best_confidence = 0

            for row_idx, row in enumerate(grid.bubbles):
                if col_idx < len(row):
                    bubble = row[col_idx]
                    if bubble.is_filled and bubble.confidence > best_confidence:
                        filled_row = row_idx
                        best_confidence = bubble.confidence

            if filled_row is not None and filled_row < len(values):
                result.append(values[filled_row])
            else:
                result.append(' ')  # Empty column = space

        return ''.join(result).strip()


class BubbleTrainer:
    """
    Trains a YOLO model for bubble detection.

    Uses existing OMR images with known coordinates to generate
    training data automatically.
    """

    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_training_data(
        self,
        image_paths: List[str],
        template_config: Dict,
        aligned_images_dir: Optional[str] = None
    ):
        """
        Generate YOLO training data from existing OMR images.

        Uses the template configuration to identify bubble positions,
        then analyzes each bubble to determine if filled.

        Args:
            image_paths: List of OMR image paths
            template_config: Template with bubble positions
            aligned_images_dir: If provided, use pre-aligned images
        """
        images_dir = self.output_dir / "images"
        labels_dir = self.output_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        for img_path in image_paths:
            self._process_training_image(
                img_path, template_config,
                images_dir, labels_dir
            )

    def _process_training_image(
        self,
        image_path: str,
        template_config: Dict,
        images_dir: Path,
        labels_dir: Path
    ):
        """Process a single image for training data."""
        img = cv2.imread(image_path)
        if img is None:
            return

        # Get image dimensions
        h, w = img.shape[:2]

        # Generate labels for each bubble position
        labels = []

        for block_name, block_config in template_config.get('fieldBlocks', {}).items():
            origin = block_config.get('origin', [0, 0])
            bubbles_gap = block_config.get('bubblesGap', 50)
            labels_gap = block_config.get('labelsGap', 50)
            bubble_values = block_config.get('bubbleValues', [])
            field_labels = block_config.get('fieldLabels', [])
            direction = block_config.get('direction', 'vertical')

            for col_idx, _ in enumerate(field_labels):
                for row_idx, _ in enumerate(bubble_values):
                    if direction == 'vertical':
                        bx = origin[0] + col_idx * labels_gap
                        by = origin[1] + row_idx * bubbles_gap
                    else:
                        bx = origin[0] + row_idx * bubbles_gap
                        by = origin[1] + col_idx * labels_gap

                    # Extract bubble region
                    bubble_size = 28  # From template
                    x1, y1 = bx, by
                    x2, y2 = bx + bubble_size, by + bubble_size

                    if x2 <= w and y2 <= h:
                        bubble_region = img[y1:y2, x1:x2]
                        is_filled = self._is_bubble_filled(bubble_region)

                        # YOLO format: class x_center y_center width height (normalized)
                        x_center = (x1 + x2) / 2 / w
                        y_center = (y1 + y2) / 2 / h
                        norm_w = bubble_size / w
                        norm_h = bubble_size / h
                        cls = 1 if is_filled else 0

                        labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        # Save image and labels
        img_name = Path(image_path).stem
        cv2.imwrite(str(images_dir / f"{img_name}.jpg"), img)

        with open(labels_dir / f"{img_name}.txt", 'w') as f:
            f.write('\n'.join(labels))

    def _is_bubble_filled(self, bubble_region: np.ndarray) -> bool:
        """Determine if a bubble region is filled."""
        if bubble_region.size == 0:
            return False

        gray = cv2.cvtColor(bubble_region, cv2.COLOR_BGR2GRAY) if len(bubble_region.shape) == 3 else bubble_region
        mean_val = np.mean(gray)

        # Filled bubbles are darker (lower mean)
        return mean_val < 150

    def create_dataset_yaml(self, train_split: float = 0.8):
        """Create YOLO dataset configuration file."""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'empty_bubble',
                1: 'filled_bubble'
            }
        }

        with open(self.output_dir / 'dataset.yaml', 'w') as f:
            import yaml
            yaml.dump(config, f)


class AdaptiveBubbleClassifier:
    """
    Adaptive bubble classifier with per-image threshold calibration.

    Key features:
    1. Uses ML classifier when available, with adaptive confidence threshold
    2. Falls back to relative darkness comparison
    3. Implements "first-filled-wins" logic for multi-marked columns
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the adaptive classifier.

        Args:
            model_path: Path to trained bubble classifier model (.pt file)
        """
        self.ml_classifier = None
        self.device = 'cpu'

        if model_path:
            self._load_ml_classifier(model_path)

    def _load_ml_classifier(self, model_path: str):
        """Load PyTorch bubble classifier model."""
        try:
            import torch
            from pathlib import Path

            if not Path(model_path).exists():
                return

            # Load PyTorch model directly
            # BubbleClassifier is a nn.Module, load it with torch.load()
            try:
                from .bubble_classifier import BubbleClassifier
                # Load state dict into new model instance
                self.ml_classifier = BubbleClassifier()
                state_dict = torch.load(model_path, map_location='cpu')
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    self.ml_classifier.load_state_dict(state_dict['model_state_dict'])
                elif isinstance(state_dict, dict):
                    self.ml_classifier.load_state_dict(state_dict)
                else:
                    # state_dict is actually the full model
                    self.ml_classifier = state_dict
                self.ml_classifier.eval()
            except ImportError:
                # Load as raw PyTorch model
                self.ml_classifier = torch.load(model_path, map_location='cpu')
                if hasattr(self.ml_classifier, 'eval'):
                    self.ml_classifier.eval()

            # Set device
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'

        except Exception as e:
            print(f"Could not load ML classifier: {e}")
            self.ml_classifier = None

    def calculate_adaptive_threshold(
        self,
        bubble_confidences: List[float]
    ) -> float:
        """
        Calculate adaptive threshold based on confidence distribution.

        For images with clear filled/empty separation, use lower threshold.
        For ambiguous images, use moderate threshold (not too strict).

        Optimized based on analysis of 302 test images - lower thresholds
        to catch light scan fills while still filtering obvious noise.

        Args:
            bubble_confidences: List of ML confidence scores for all bubbles

        Returns:
            Adaptive threshold value (0.4-0.7 range)
        """
        if not bubble_confidences:
            return 0.5  # Lower default

        confidences = np.array(bubble_confidences)
        std_conf = np.std(confidences)

        # High variance = clear separation = lower threshold OK
        if std_conf > 0.25:
            return 0.4  # Very clear separation
        elif std_conf > 0.18:
            return 0.45
        elif std_conf > 0.12:
            return 0.5
        elif std_conf > 0.08:
            return 0.55
        else:
            # Low variance = still be permissive
            return 0.6

    def classify_bubbles_in_column(
        self,
        bubble_images: List[np.ndarray],
        threshold: Optional[float] = None,
        use_threshold_fallback: bool = True,
        strict_mode: Optional[bool] = None
    ) -> List[Tuple[bool, float]]:
        """
        Classify all bubbles in a column using hybrid ML + threshold approach.

        Args:
            bubble_images: List of bubble images (grayscale, ~28x28)
            threshold: Optional explicit threshold, or auto-calculate
            use_threshold_fallback: If True, use threshold-based as fallback/tiebreaker
            strict_mode: If True, use stricter thresholds. If None, auto-detect
                        based on number of bubbles (26 = name field = strict)

        Returns:
            List of (is_filled, confidence) tuples
        """
        if not bubble_images:
            return []

        # Auto-detect strict mode: 26 bubbles = name field = use strict thresholds
        if strict_mode is None:
            strict_mode = len(bubble_images) >= 26

        # Always compute threshold-based results as fallback
        threshold_results = self._classify_threshold_batch(bubble_images, strict_mode=strict_mode)

        if self.ml_classifier is not None:
            # Use ML classifier
            ml_results = self._classify_ml_batch(bubble_images)

            # Calculate adaptive threshold if not provided
            if threshold is None:
                confidences = [conf for _, conf in ml_results]
                threshold = self.calculate_adaptive_threshold(confidences)

            results = []
            for i, (is_filled, conf) in enumerate(ml_results):
                # Apply adaptive threshold
                if is_filled and conf >= threshold:
                    results.append((True, conf))
                elif not is_filled and (1 - conf) >= threshold:
                    results.append((False, 1 - conf))
                else:
                    # Ambiguous ML result - be conservative
                    # Only use threshold as tiebreaker if threshold has high confidence (>0.7)
                    if use_threshold_fallback and i < len(threshold_results):
                        thresh_is_filled, thresh_conf = threshold_results[i]
                        if thresh_conf >= 0.7:
                            results.append(threshold_results[i])
                        else:
                            # Both ML and threshold are ambiguous - default to not filled
                            results.append((False, 0.5))
                    else:
                        results.append((is_filled, conf))

            # If ML finds no filled bubbles but threshold does, use threshold results
            ml_filled = sum(1 for is_f, _ in results if is_f)
            thresh_filled = sum(1 for is_f, _ in threshold_results if is_f)

            if ml_filled == 0 and thresh_filled > 0:
                return threshold_results

            return results
        else:
            # No ML classifier - use threshold-based
            return threshold_results

    def _classify_ml_batch(
        self,
        bubble_images: List[np.ndarray]
    ) -> List[Tuple[bool, float]]:
        """Classify bubbles using ML model."""
        try:
            import torch

            results = []
            for img in bubble_images:
                # Preprocess
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Resize to expected size
                img = cv2.resize(img, (28, 28))

                # Normalize and convert to tensor
                tensor = torch.FloatTensor(img).unsqueeze(0).unsqueeze(0) / 255.0

                # Move to device if needed
                if hasattr(self.ml_classifier, 'eval'):
                    self.ml_classifier.eval()
                    with torch.no_grad():
                        output = self.ml_classifier(tensor)
                        prob = torch.sigmoid(output).item()
                        is_filled = prob > 0.5
                        confidence = prob if is_filled else (1 - prob)
                        results.append((is_filled, confidence))
                elif hasattr(self.ml_classifier, 'predict'):
                    is_filled, confidence = self.ml_classifier.predict(img)
                    results.append((is_filled, confidence))
                else:
                    results.append((False, 0.5))

            return results

        except Exception as e:
            # Fallback to threshold
            return self._classify_threshold_batch(bubble_images)

    def _classify_threshold_batch(
        self,
        bubble_images: List[np.ndarray],
        strict_mode: bool = False
    ) -> List[Tuple[bool, float]]:
        """
        Classify bubbles using darkness threshold with relative ratio.

        Args:
            bubble_images: List of bubble images
            strict_mode: If True, use stricter thresholds for name fields (25+ columns)
                        to avoid false positives in empty columns

        Optimized thresholds based on visual analysis of 302 test images:
        1. Strong relative (1.5x median) with minimum absolute - catches light scans
        2. Moderate relative (1.3x median) with good absolute - balanced
        3. High absolute with gap - catches dark scans

        This handles varying scan quality while preventing false positives.
        """
        results = []
        darkness_values = []

        # Calculate darkness for all bubbles
        for img in bubble_images:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            darkness = 255 - np.mean(img)  # Higher = darker (filled)
            darkness_values.append(darkness)

        if not darkness_values:
            return results

        # Calculate statistics
        median_darkness = np.median(darkness_values)
        max_darkness = max(darkness_values)
        min_darkness = min(darkness_values)

        # Use max(median, 5) to avoid division issues with very low baselines
        safe_median = max(median_darkness, 5)

        for i, darkness in enumerate(darkness_values):
            # Calculate relative ratio and gap
            relative_ratio = darkness / safe_median
            gap = darkness - median_darkness

            # Relative gap: what fraction of the max-median range is this bubble's gap?
            max_gap = max_darkness - median_darkness
            relative_gap = gap / max(max_gap, 1)

            # Detection criteria (optimized 2026-01-03):
            # When there's a clearly filled bubble (max > 150), be stricter about what counts as filled
            # This prevents false positives from slight variations when actual filled bubble is obvious
            is_filled = False

            if max_darkness > 150:
                # High contrast image - use stricter criteria
                # Only mark as filled if darkness is at least 40% of max or relative_gap > 0.5
                if darkness >= max_darkness * 0.4 or (darkness >= 100 and relative_gap > 0.5):
                    if darkness >= 12 and relative_ratio >= 1.5:
                        is_filled = True
                    elif darkness >= 25 and relative_ratio >= 1.3:
                        is_filled = True
            else:
                # Low contrast image
                if strict_mode:
                    # Stricter criteria for name fields to avoid false positives
                    if darkness >= 50 and relative_ratio >= 1.5:
                        is_filled = True
                    elif darkness >= 60 and relative_ratio >= 1.3:
                        is_filled = True
                    elif darkness >= 80 and gap >= 15:
                        is_filled = True
                else:
                    # Original criteria for section/class/roll fields
                    if darkness >= 12 and relative_ratio >= 1.5:
                        is_filled = True
                    elif darkness >= 25 and relative_ratio >= 1.3:
                        is_filled = True
                    elif darkness >= 35 and gap >= 8:
                        is_filled = True

            if is_filled:
                # Likely filled - confidence based on how much it stands out
                confidence = min(1.0, 0.5 + gap / (max_darkness - median_darkness + 1) * 0.5)
                results.append((True, confidence))
            else:
                # Likely empty
                confidence = min(1.0, 0.5 + (median_darkness - darkness) / (median_darkness - min_darkness + 1) * 0.5)
                results.append((False, confidence))

        return results

    def select_from_column(
        self,
        bubble_results: List[Tuple[int, bool, float]]
    ) -> Tuple[Optional[int], float]:
        """
        Select which bubble is filled in a column.

        IMPORTANT: When multiple bubbles are filled, picks FIRST one (lowest row index),
        NOT the darkest one. This matches the user requirement.

        Args:
            bubble_results: List of (row_index, is_filled, confidence) tuples

        Returns:
            (selected_row_index, confidence) or (None, 0.0) if no bubble is filled
        """
        # Filter to filled bubbles
        filled = [(row_idx, conf) for row_idx, is_filled, conf in bubble_results if is_filled]

        if not filled:
            return (None, 0.0)

        if len(filled) == 1:
            return filled[0]

        # Multiple filled: pick FIRST (lowest row index)
        filled.sort(key=lambda x: x[0])  # Sort by row index ascending
        return filled[0]

    def classify_and_select(
        self,
        bubble_images: List[np.ndarray],
        values: List[str]
    ) -> Tuple[str, float]:
        """
        Classify bubbles in a column and return the selected value.

        Args:
            bubble_images: List of bubble images for each row in the column
            values: Possible values for each row (e.g., ['A', 'B', ..., 'Z'])

        Returns:
            (selected_value, confidence) or (' ', 0.0) if no selection
        """
        results = self.classify_bubbles_in_column(bubble_images)

        # Create bubble_results with row indices
        bubble_results = [
            (row_idx, is_filled, conf)
            for row_idx, (is_filled, conf) in enumerate(results)
        ]

        selected_row, confidence = self.select_from_column(bubble_results)

        if selected_row is not None and selected_row < len(values):
            return (values[selected_row], confidence)
        else:
            return (' ', 0.0)


def create_adaptive_classifier(model_path: Optional[str] = None) -> AdaptiveBubbleClassifier:
    """
    Factory function to create an AdaptiveBubbleClassifier.

    Searches for model in common locations if path not provided.
    """
    if model_path is None:
        # Try common model locations
        common_paths = [
            "models/bubble_classifier_v4.pt",
            "models/bubble_classifier_v3.pt",
            "models/bubble_classifier_v2.pt",
            "models/bubble_classifier.pt",
        ]
        for path in common_paths:
            if Path(path).exists():
                model_path = path
                break

    return AdaptiveBubbleClassifier(model_path)

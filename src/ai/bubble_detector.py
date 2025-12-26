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

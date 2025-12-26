"""
Pure YOLOv8-based OMR processor.

Uses YOLOv8 object detection to directly detect filled bubbles
without relying on fixed coordinate-based grid sampling.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json


@dataclass
class BubbleDetection:
    """A single detected bubble."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    is_filled: bool
    grid_row: int = -1  # A=0, B=1, ..., Z=25
    grid_col: int = -1  # Column position (0-24 for name)


@dataclass
class OMRResult:
    """Result of OMR processing."""
    success: bool
    name: str = ""
    roll_number: str = ""
    class_: str = ""
    section: str = ""
    school_code: str = ""
    answers: Dict[int, str] = field(default_factory=dict)
    confidence: float = 0.0
    error_message: str = ""
    all_detections: List[BubbleDetection] = field(default_factory=list)


class YOLOOMRProcessor:
    """
    Process OMR sheets using YOLOv8 for bubble detection.

    Two modes:
    1. Trained model: Uses a trained YOLOv8 model to detect filled bubbles
    2. Zero-shot: Uses image processing to detect and classify bubbles
    """

    # Grid definitions for Vidya Bharti 2025 OMR sheet
    # These define regions and expected grid structure
    GRID_REGIONS = {
        'name': {
            'region': (100, 600, 1500, 2100),  # (x, y, x2, y2)
            'rows': 26,  # A-Z
            'cols': 25,
            'values': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            'expected_bubble_size': (28, 28)
        },
        'roll_number': {
            'region': (1550, 1200, 1850, 1800),
            'rows': 10,  # 0-9
            'cols': 3,
            'values': list('0123456789'),
            'expected_bubble_size': (28, 28)
        },
        'class': {
            'region': (1750, 1200, 1850, 1700),
            'rows': 8,  # 3-10
            'cols': 1,
            'values': ['3', '4', '5', '6', '7', '8', '9', '10'],
            'expected_bubble_size': (28, 28)
        },
        'section': {
            'region': (1800, 1200, 1900, 1700),
            'rows': 8,  # A-H
            'cols': 1,
            'values': list('ABCDEFGH'),
            'expected_bubble_size': (28, 28)
        },
        'school_code': {
            'region': (1850, 1200, 2100, 1800),
            'rows': 10,  # 0-9
            'cols': 3,
            'values': list('0123456789'),
            'expected_bubble_size': (28, 28)
        }
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = False,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the YOLO OMR processor.

        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
            use_gpu: Whether to use GPU for inference
            confidence_threshold: Minimum confidence for detections
        """
        self.model = None
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """Load YOLOv8 model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            if self.use_gpu:
                self.model.to('cuda')
            print(f"Loaded YOLO model: {model_path}")
        except ImportError:
            print("ultralytics not installed. Install with: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def process(self, image_path: str) -> OMRResult:
        """
        Process an OMR sheet image.

        Args:
            image_path: Path to the OMR image

        Returns:
            OMRResult with extracted data
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return OMRResult(
                    success=False,
                    error_message=f"Failed to load image: {image_path}"
                )

            # Detect all filled bubbles
            if self.model:
                detections = self._detect_with_yolo(image)
            else:
                detections = self._detect_with_cv(image)

            # Map detections to grid positions
            result = self._map_detections_to_fields(detections, image.shape)
            result.all_detections = detections

            return result

        except Exception as e:
            import traceback
            return OMRResult(
                success=False,
                error_message=f"Processing error: {str(e)}\n{traceback.format_exc()}"
            )

    def _detect_with_yolo(self, image: np.ndarray) -> List[BubbleDetection]:
        """Detect filled bubbles using YOLOv8 model."""
        results = self.model(image, conf=self.confidence_threshold)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Class 1 = filled bubble, Class 0 = empty
                is_filled = cls == 1

                if is_filled:  # Only keep filled bubbles
                    detections.append(BubbleDetection(
                        x=int(x1),
                        y=int(y1),
                        width=int(x2 - x1),
                        height=int(y2 - y1),
                        confidence=conf,
                        is_filled=True
                    ))

        return detections

    def _detect_with_cv(self, image: np.ndarray) -> List[BubbleDetection]:
        """
        Detect filled bubbles using traditional CV when no YOLO model is available.

        This scans for dark circular/oval regions that indicate filled bubbles.
        """
        detections = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to find dark regions
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 10
        )

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size (bubbles are typically 20-40 pixels)
            if 15 < w < 50 and 15 < h < 50:
                # Check aspect ratio (bubbles are roughly circular)
                aspect_ratio = w / h
                if 0.7 < aspect_ratio < 1.4:
                    # Check if the region is dark enough (filled)
                    bubble_region = gray[y:y+h, x:x+w]
                    mean_intensity = np.mean(bubble_region)

                    # Filled bubbles are dark (low intensity)
                    if mean_intensity < 100:
                        # Calculate fill percentage
                        contour_area = cv2.contourArea(contour)
                        bbox_area = w * h
                        fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0

                        if fill_ratio > 0.4:  # Must be reasonably filled
                            detections.append(BubbleDetection(
                                x=x,
                                y=y,
                                width=w,
                                height=h,
                                confidence=1.0 - (mean_intensity / 255),
                                is_filled=True
                            ))

        return detections

    def _map_detections_to_fields(
        self,
        detections: List[BubbleDetection],
        image_shape: Tuple[int, int, int]
    ) -> OMRResult:
        """Map detected bubbles to form fields based on their position."""

        if not detections:
            return OMRResult(
                success=False,
                error_message="No filled bubbles detected"
            )

        result = OMRResult(success=True)

        # Process each field region
        for field_name, config in self.GRID_REGIONS.items():
            region = config['region']
            values = config['values']
            num_cols = config['cols']
            num_rows = config['rows']

            # Find detections within this region
            field_detections = [
                d for d in detections
                if region[0] <= d.x <= region[2] and region[1] <= d.y <= region[3]
            ]

            if field_detections:
                # Assign grid positions based on relative position in region
                field_value = self._extract_field_value(
                    field_detections, region, num_rows, num_cols, values
                )

                # Set the appropriate result field
                if field_name == 'name':
                    result.name = field_value
                elif field_name == 'roll_number':
                    result.roll_number = field_value
                elif field_name == 'class':
                    result.class_ = field_value
                elif field_name == 'section':
                    result.section = field_value
                elif field_name == 'school_code':
                    result.school_code = field_value

        # Calculate confidence
        total_conf = sum(d.confidence for d in detections)
        result.confidence = total_conf / len(detections) if detections else 0.0

        return result

    def _extract_field_value(
        self,
        detections: List[BubbleDetection],
        region: Tuple[int, int, int, int],
        num_rows: int,
        num_cols: int,
        values: List[str]
    ) -> str:
        """Extract field value from detections within a region."""

        x1, y1, x2, y2 = region
        region_width = x2 - x1
        region_height = y2 - y1

        # Calculate grid cell sizes
        col_size = region_width / num_cols if num_cols > 0 else region_width
        row_size = region_height / num_rows if num_rows > 0 else region_height

        # Map each detection to a grid cell
        for det in detections:
            # Calculate center of detection
            center_x = det.x + det.width / 2
            center_y = det.y + det.height / 2

            # Calculate grid position
            det.grid_col = int((center_x - x1) / col_size)
            det.grid_row = int((center_y - y1) / row_size)

            # Clamp to valid range
            det.grid_col = max(0, min(num_cols - 1, det.grid_col))
            det.grid_row = max(0, min(num_rows - 1, det.grid_row))

        # Build result by column
        result_chars = []
        for col in range(num_cols):
            # Find detections in this column
            col_detections = [d for d in detections if d.grid_col == col]

            if col_detections:
                # Take the detection with highest confidence, or first row if tie
                best = min(col_detections, key=lambda d: (d.grid_row, -d.confidence))
                if best.grid_row < len(values):
                    result_chars.append(values[best.grid_row])
                else:
                    result_chars.append(' ')
            else:
                result_chars.append(' ')

        return ''.join(result_chars).strip()


class YOLOTrainer:
    """
    Generate training data and train YOLOv8 model for OMR bubble detection.
    """

    def __init__(self, output_dir: str = "yolo_training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_training_data(
        self,
        image_paths: List[str],
        reference_image_path: Optional[str] = None,
        sample_count: int = 50
    ):
        """
        Generate YOLO training data from OMR images.

        Uses a semi-automated approach:
        1. Detect potential bubble locations using CV
        2. Classify each as filled/empty based on darkness
        3. Save in YOLO format

        Args:
            image_paths: List of OMR image paths
            reference_image_path: Optional reference for alignment
            sample_count: Number of images to process
        """
        images_dir = self.output_dir / "images" / "train"
        labels_dir = self.output_dir / "labels" / "train"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        processed = 0
        for img_path in image_paths[:sample_count]:
            if self._process_image_for_training(img_path, images_dir, labels_dir):
                processed += 1
                print(f"Processed {processed}/{sample_count}: {Path(img_path).name}")

        # Create dataset.yaml
        self._create_dataset_yaml()

        print(f"\nTraining data generated: {processed} images")
        print(f"Output directory: {self.output_dir}")

    def _process_image_for_training(
        self,
        image_path: str,
        images_dir: Path,
        labels_dir: Path
    ) -> bool:
        """Process a single image for training."""
        img = cv2.imread(image_path)
        if img is None:
            return False

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find all potential bubbles using Hough circles
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=25,
            minRadius=10,
            maxRadius=25
        )

        labels = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0]:
                cx, cy, r = circle

                # Extract bubble region
                x1 = max(0, cx - r)
                y1 = max(0, cy - r)
                x2 = min(w, cx + r)
                y2 = min(h, cy + r)

                bubble_region = gray[y1:y2, x1:x2]
                if bubble_region.size == 0:
                    continue

                # Determine if filled based on mean darkness
                mean_val = np.mean(bubble_region)
                is_filled = mean_val < 100  # Dark = filled

                # YOLO format: class x_center y_center width height (normalized)
                x_center = cx / w
                y_center = cy / h
                bubble_w = (2 * r) / w
                bubble_h = (2 * r) / h
                cls = 1 if is_filled else 0

                labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {bubble_w:.6f} {bubble_h:.6f}")

        if labels:
            # Save image and labels
            img_name = Path(image_path).stem
            cv2.imwrite(str(images_dir / f"{img_name}.jpg"), img)
            with open(labels_dir / f"{img_name}.txt", 'w') as f:
                f.write('\n'.join(labels))
            return True

        return False

    def _create_dataset_yaml(self):
        """Create YOLO dataset configuration."""
        yaml_content = f"""
path: {self.output_dir.absolute()}
train: images/train
val: images/train  # Using same for now, split manually for production

names:
  0: empty_bubble
  1: filled_bubble
"""
        with open(self.output_dir / "dataset.yaml", 'w') as f:
            f.write(yaml_content)

    def train(self, epochs: int = 100, batch_size: int = 16):
        """
        Train YOLOv8 model on generated data.

        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        try:
            from ultralytics import YOLO

            # Start with YOLOv8 nano (smallest, fastest)
            model = YOLO('yolov8n.pt')

            # Train on our dataset
            results = model.train(
                data=str(self.output_dir / "dataset.yaml"),
                epochs=epochs,
                batch=batch_size,
                imgsz=640,
                project=str(self.output_dir / "runs"),
                name="omr_bubble_detector"
            )

            print(f"\nTraining complete!")
            print(f"Best model saved to: {self.output_dir / 'runs' / 'omr_bubble_detector' / 'weights' / 'best.pt'}")

            return results

        except ImportError:
            print("ultralytics not installed. Install with: pip install ultralytics")
            return None


def test_processor():
    """Test the YOLO OMR processor."""
    print("Testing YOLO OMR Processor")
    print("=" * 60)

    # Initialize without model (will use CV detection)
    processor = YOLOOMRProcessor()

    # Test on sample images
    test_images = list(Path("filledomrsheet").glob("*.jpg"))[:5]

    for img_path in test_images:
        result = processor.process(str(img_path))
        print(f"\n{img_path.name}:")
        print(f"  Name: {result.name}")
        print(f"  Roll: {result.roll_number}")
        print(f"  Class: {result.class_}")
        print(f"  Section: {result.section}")
        print(f"  Detections: {len(result.all_detections)}")


if __name__ == "__main__":
    test_processor()

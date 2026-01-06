"""
Pure YOLOv8-based OMR processor.

Uses YOLOv8 object detection to directly detect filled bubbles
without relying on fixed coordinate-based grid sampling.

Now includes ML-enhanced answer detection using a trained bubble classifier.

NEW: Supports modular pipeline with marker-based alignment and dynamic grid detection.
Set USE_NEW_PIPELINE = True to use the new robust detection system.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

# Feature flag for new modular pipeline
# Set to True to use marker-based alignment and dynamic grid detection
# Calibrated 2025-01-02 with correct grid positions in canonical frame (2480x3507)
USE_NEW_PIPELINE = True

# Try to import torch for ML classifier
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # Placeholder

# Import new modular components (for USE_NEW_PIPELINE)
try:
    from .marker_aligner import MarkerAligner, AlignmentResult
    from .grid_detector import GridDetector, GridConfig, GRID_SEARCH_REGIONS
    from .bubble_detector import AdaptiveBubbleClassifier, create_adaptive_classifier
    from .field_extractor import FieldExtractor, OMRExtractionResult
    NEW_MODULES_AVAILABLE = True
except ImportError as e:
    NEW_MODULES_AVAILABLE = False
    print(f"New modules not available: {e}")


# Only define CNN class if torch is available
if TORCH_AVAILABLE:
    class AnswerBubbleCNN(nn.Module):
        """CNN for answer bubble classification (empty vs A/B/C/D)."""

        def __init__(self, num_classes=5):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(32, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))


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
    ocr_used: bool = False  # True if OCR fallback was used for name
    field_confidences: Dict[str, float] = field(default_factory=dict)  # Per-field confidence


class YOLOOMRProcessor:
    """
    Process OMR sheets using YOLOv8 for bubble detection.

    Two modes:
    1. Trained model: Uses a trained YOLOv8 model to detect filled bubbles
    2. Grid-based: Uses direct grid sampling at known coordinates
    """

    # Target dimensions for alignment (matches template.json pageDimensions)
    TARGET_WIDTH = 2480
    TARGET_HEIGHT = 3507

    # Grid definitions for Vidya Bharti 2025 OMR sheet
    # ORIGINAL coordinates measured on 1662x2320 image using pixspy
    # These are scaled dynamically based on each image's actual dimensions
    # using: scaled_coord = original_coord * (TARGET_SIZE / original_image_size)
    FIELD_CONFIGS_ORIGINAL = {
        # Original pixspy coordinates measured on 1662x2320 image (41_625.jpg)
        # User provided: roll(1067,857), class(1212,857), school(1424,857)
        # Adjusted: X-10, Y-10 + width-based X offset for narrower images
        'roll_number': {
            'origin': (1057, 847),  # Adjusted from (1067, 857)
            'bubble_size': (19, 19),
            'rows_gap': 35,
            'cols_gap': 44,
            'rows': 10,
            'cols': 3,
            'values': list('0123456789'),
            'direction': 'vertical'
        },
        'class': {
            'origin': (1202, 847),  # Adjusted from (1212, 857)
            'bubble_size': (19, 19),
            'rows_gap': 35,
            'cols_gap': 40,
            'rows': 8,
            'cols': 1,
            'values': ['3', '4', '5', '6', '7', '8', '9', '10'],
            'direction': 'vertical'
        },
        'section': {
            'origin': (1308, 847),  # Adjusted from (1318, 857)
            'bubble_size': (19, 19),
            'rows_gap': 35,
            'cols_gap': 40,
            'rows': 8,
            'cols': 1,
            'values': list('ABCDEFGH'),
            'direction': 'vertical'
        },
        'school_code': {
            'origin': (1414, 847),  # Adjusted from (1424, 857)
            'bubble_size': (19, 19),
            'rows_gap': 35,
            'cols_gap': 41,
            'rows': 10,
            'cols': 3,
            'values': list('0123456789'),
            'direction': 'vertical'
        },
        'name': {
            # Calibrated from pixspy on 41_625.jpg:
            # Col0 A center: (129,465), Col24 A center: (990,470)
            # Col0 Z center: (129,1315), Col24 Z center: (989,1316)
            'origin': (128, 454),  # Moved right by 10px for better alignment
            'bubble_size': (22, 22),
            'rows_gap': 34,  # (1315-465)/25 = 34
            'cols_gap': 36,  # (990-129)/24 = 35.875 ≈ 36
            'x_shift_per_row': 0,  # No shift - bubbles are vertically straight
            'y_drift_per_col': 0.0,  # Set to 0 to fix col24 Z alignment
            'x_drift_per_col': 0.3,  # Compensate for leftward drift across columns
            'y_drift_per_row': 0.3,  # Small compensation for upward drift across rows
            'rows': 26,
            'cols': 25,
            'values': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            'direction': 'vertical'
        }
    }
    # Pre-scaled configs (for backward compatibility with non-dynamic code)
    FIELD_CONFIGS = {
        'name': {
            'origin': (243, 690),
            'origin_chayan': (235, 660),  # Calibrated for Chayan blue forms
            'bubble_size': (33, 33),
            'rows_gap': 51,
            'cols_gap': 51,  # Calibrated: 49-53px works, using 51
            'x_shift_per_row': -3.56,
            'rows': 26,
            'cols': 25,
            'values': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            'direction': 'vertical',
            # Chayan forms have extra gaps between name sections (first/middle/last name)
            # After col 5 (end of first name) and col 10 (end of middle name)
            # Gap is 99px instead of standard 51px, so extra 48px
            'extra_gaps_chayan': {5: 48, 10: 48},  # col_idx: extra_pixels
        },
        'roll_number': {
            'origin': (1640, 1289),
            'bubble_size': (28, 28),
            'rows_gap': 53,
            'cols_gap': 66,
            'rows': 10,
            'cols': 3,
            'values': list('0123456789'),
            'direction': 'vertical'
        },
        'class': {
            'origin': (1790, 1289),
            'bubble_size': (28, 28),
            'rows_gap': 53,
            'cols_gap': 60,
            'rows': 8,
            'cols': 1,
            'values': ['3', '4', '5', '6', '7', '8', '9', '10'],
            'direction': 'vertical'
        },
        'section': {
            'origin': (1951, 1289),
            'bubble_size': (28, 28),
            'rows_gap': 53,
            'cols_gap': 60,
            'rows': 8,
            'cols': 1,
            'values': list('ABCDEFGH'),
            'direction': 'vertical'
        },
        'school_code': {
            'origin': (2112, 1289),
            'bubble_size': (28, 28),
            'rows_gap': 53,
            'cols_gap': 62,
            'rows': 10,
            'cols': 3,
            'values': list('0123456789'),
            'direction': 'vertical'
        }
    }

    # Answer blocks configuration
    # Answer section: 6 blocks, each with 10 questions arranged VERTICALLY
    # Options A,B,C,D go HORIZONTALLY within each question
    #
    # ORIGINAL coordinates measured on 1662x2320 image (41_625.jpg) using pixspy:
    # Q1 Option A: (186, 1459), Q10 Option A: (186, 2006)
    # Q11 Option A: (425, 1459), Q51 Option A: (1380, 1461)
    # Row gap: (2006 - 1459) / 9 = 60.78 ≈ 61 pixels
    # Block spacing: 425 - 186 = 239 pixels
    ANSWER_BLOCKS_ORIGINAL = {
        'blocks': [
            # Shifted LEFT by 43px (one option_gap) to correct column alignment
            {'origin': (128, 1449), 'questions': range(1, 11)},    # Block 1: Q1-10 (was 171)
            {'origin': (367, 1449), 'questions': range(11, 21)},   # Block 2: Q11-20 (was 410)
            {'origin': (606, 1449), 'questions': range(21, 31)},   # Block 3: Q21-30 (was 649)
            {'origin': (845, 1449), 'questions': range(31, 41)},   # Block 4: Q31-40 (was 888)
            {'origin': (1084, 1449), 'questions': range(41, 51)},  # Block 5: Q41-50 (was 1127)
            {'origin': (1322, 1451), 'questions': range(51, 61)},  # Block 6: Q51-60 (was 1365)
        ],
        'row_gap': 61,       # (2006 - 1459) / 9 = 60.78
        'option_gap': 43,    # Horizontal gap between A,B,C,D (estimated from pixspy)
        'bubble_size': (19, 19),  # Same as other fields on original image
    }

    # Pre-scaled answer blocks (legacy, kept for reference)
    ANSWER_BLOCKS = [
        {'origin': (261, 2111), 'questions': range(1, 11)},    # Block 1: Q1-10
        {'origin': (607, 2111), 'questions': range(11, 21)},   # Block 2: Q11-20 (+346)
        {'origin': (953, 2111), 'questions': range(21, 31)},   # Block 3: Q21-30 (+346)
        {'origin': (1299, 2111), 'questions': range(31, 41)},  # Block 4: Q31-40 (+346)
        {'origin': (1645, 2111), 'questions': range(41, 51)},  # Block 5: Q41-50 (+346)
        {'origin': (1991, 2111), 'questions': range(51, 61)},  # Block 6: Q51-60 (+346)
    ]
    # Chayan batch (39_*.jpg) has different alignment
    ANSWER_BLOCKS_CHAYAN = [
        {'origin': (252, 2112), 'questions': range(1, 11)},    # Block 1: Q1-10
        {'origin': (612, 2112), 'questions': range(11, 21)},   # Block 2: Q11-20
        {'origin': (972, 2112), 'questions': range(21, 31)},   # Block 3: Q21-30
        {'origin': (1332, 2112), 'questions': range(31, 41)},  # Block 4: Q31-40
        {'origin': (1692, 2112), 'questions': range(41, 51)},  # Block 5: Q41-50
        {'origin': (2052, 2112), 'questions': range(51, 61)},  # Block 6: Q51-60
    ]
    ANSWER_BUBBLE_SIZE = (28, 28)
    ANSWER_OPTION_GAP = 65   # Horizontal gap between A,B,C,D options (was 70)
    # Row positions calibrated from actual filled bubbles after marker alignment
    # Gaps are approximately 104 pixels between rows
    # Row positions for Khoj batch (TOP-LEFT of bubbles)
    # IMPORTANT: First position is the LABEL ROW (A B C D markers), NOT Q1!
    # Calibrated from peak detection: Q1 center ≈2171, gap ≈104px
    ANSWER_ROW_POSITIONS = [
        2055,  # Label row (A B C D markers)
        2157,  # Q1 (center 2171 - 14)
        2261,  # Q2 (+104)
        2365,  # Q3 (+104)
        2469,  # Q4 (+104)
        2573,  # Q5 (+104)
        2677,  # Q6 (+104)
        2781,  # Q7 (+104)
        2885,  # Q8 (+104)
        2989,  # Q9 (+104)
        3093,  # Q10 (+104)
    ]
    # Chayan batch row positions (TOP-LEFT of bubbles, not centers)
    # IMPORTANT: First position is the LABEL ROW (A B C D markers), NOT Q1!
    # Detected centers: Q1≈2126, Q2≈2207, Q3≈2299; subtract 14 for top-left
    ANSWER_ROW_POSITIONS_CHAYAN = [
        2112,  # Label row (A B C D markers)
        2193,  # Q1 (center 2207 - 14)
        2285,  # Q2 (center 2299 - 14)
        2377,  # Q3
        2469,  # Q4
        2561,  # Q5
        2653,  # Q6
        2745,  # Q7
        2837,  # Q8
        2929,  # Q9
        3021,  # Q10 (estimated, ~92px gap from Q9)
    ]
    ANSWER_VALUES = list('ABCD')

    def __init__(
        self,
        model_path: Optional[str] = None,
        reference_image_path: Optional[str] = None,
        use_gpu: bool = False,
        confidence_threshold: float = 0.5,
        bubble_classifier_path: Optional[str] = None,
        answer_classifier_path: Optional[str] = None
    ):
        """
        Initialize the YOLO OMR processor.

        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
            reference_image_path: Path to reference image for alignment
            use_gpu: Whether to use GPU for inference
            confidence_threshold: Minimum confidence for detections
            bubble_classifier_path: Path to trained bubble classifier (.pt file)
            answer_classifier_path: Path to trained answer classifier (.pt file)
        """
        self.model = None
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self.reference_image = None
        self.bubble_classifier = None
        self.answer_classifier = None
        self.answer_classifier_device = None

        # Create instance copy of FIELD_CONFIGS to avoid cross-instance contamination
        # when calibration updates the config values
        import copy
        self.field_configs = copy.deepcopy(self.FIELD_CONFIGS)

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

        if reference_image_path and Path(reference_image_path).exists():
            self.reference_image = cv2.imread(reference_image_path)

        # Load ML bubble classifier (binary filled/empty classification)
        # Prefer v4 model trained with RELATIVE labeling (1040 filled + 12550 empty)
        if bubble_classifier_path is None:
            bubble_default_paths = [
                Path(__file__).parent.parent.parent / "models" / "bubble_classifier_v4.pt",
                Path(__file__).parent.parent.parent / "models" / "bubble_classifier_v3.pt",
                Path(__file__).parent.parent.parent / "models" / "bubble_classifier_v2.pt",
                Path(__file__).parent.parent.parent / "models" / "name_bubble_classifier.pt",
                Path("models/bubble_classifier_v4.pt"),
                Path("models/bubble_classifier_v3.pt"),
                Path("models/bubble_classifier_v2.pt"),
            ]
            for p in bubble_default_paths:
                if p.exists():
                    bubble_classifier_path = str(p)
                    break
        if bubble_classifier_path and Path(bubble_classifier_path).exists():
            self._load_bubble_classifier(bubble_classifier_path)

        # Load ML answer classifier - try default path if not specified
        # Note: answer_classifier uses AnswerBubbleCNN (5-class), not BubbleClassifier (binary)
        if answer_classifier_path is None:
            default_paths = [
                Path(__file__).parent.parent.parent / "models" / "bubble_classifier.pt",
                Path("models/bubble_classifier.pt"),
            ]
            for p in default_paths:
                if p.exists():
                    answer_classifier_path = str(p)
                    break

        if answer_classifier_path and Path(answer_classifier_path).exists():
            self._load_answer_classifier(answer_classifier_path)

        # Initialize OCR extractor for fallback
        self.ocr_extractor = None
        self._init_ocr_extractor()

    def _init_ocr_extractor(self):
        """Initialize OCR extractor for fallback name detection."""
        try:
            from src.ai.ocr_extractor import OCRNameExtractor
            self.ocr_extractor = OCRNameExtractor()
        except Exception as e:
            print(f"OCR extractor not available: {e}")
            self.ocr_extractor = None

    def _load_bubble_classifier(self, model_path: str):
        """Load the ML bubble classifier."""
        try:
            from src.ai.bubble_classifier import BubbleClassifierTrainer
            self.bubble_classifier = BubbleClassifierTrainer(model_path)
            print(f"Loaded bubble classifier: {model_path}")
        except Exception as e:
            print(f"Failed to load bubble classifier: {e}")
            self.bubble_classifier = None

    def _load_answer_classifier(self, model_path: str):
        """Load the ML answer classifier for improved answer detection."""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, skipping ML answer classifier")
            return

        try:
            # Detect device
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            self.answer_classifier_device = device

            # Load model
            self.answer_classifier = AnswerBubbleCNN(num_classes=5)
            checkpoint = torch.load(model_path, map_location=device)
            self.answer_classifier.load_state_dict(checkpoint['model_state_dict'])
            self.answer_classifier.to(device)
            self.answer_classifier.eval()

            print(f"Loaded answer classifier: {model_path} (device: {device})")
        except Exception as e:
            print(f"Failed to load answer classifier: {e}")
            self.answer_classifier = None

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
            import os
            image = cv2.imread(image_path)
            if image is None:
                return OMRResult(
                    success=False,
                    error_message=f"Failed to load image: {image_path}"
                )

            # Use new modular pipeline if enabled and available
            if USE_NEW_PIPELINE and NEW_MODULES_AVAILABLE:
                result = self._process_with_new_pipeline(image, image_path)
                # Still verify with OCR for name
                result = self._verify_name_with_ocr(result, image_path)
                return result

            # --- Legacy pipeline below ---

            # Align image to target dimensions
            aligned = self._align_image(image)

            # Auto-detect batch type by form color (Blue=Chayan, Red=Khoj)
            self._is_chayan_batch = self._detect_best_origin(aligned)

            # Default to 2025-26 form layout
            # The form year detection was unreliable - auto-calibration handles variations
            # TODO: Implement more reliable form year detection if needed
            self._is_2023_form = False

            if self._is_chayan_batch:
                # Chayan batch - use adjusted origins
                self.field_configs['name']['origin'] = self.field_configs['name'].get('origin_chayan', (180, 710))
                self._current_answer_blocks = self.ANSWER_BLOCKS_CHAYAN
                self._current_answer_row_positions = self.ANSWER_ROW_POSITIONS_CHAYAN
            elif self._is_2023_form:
                # 2023-24 form (Khoj batch) - different grid position
                # After marker alignment, adjusted for 2023-24 layout
                self.field_configs['name']['origin'] = (220, 610)  # Calibrated for 2023-24 forms
                self.field_configs['name']['rows_gap'] = 54
                self.field_configs['name']['cols_gap'] = 54
                # Calibrated origins for 2023-24 forms
                self.field_configs['roll_number']['origin'] = (1600, 1250)
                self.field_configs['class']['origin'] = (1810, 1250)
                self.field_configs['section']['origin'] = (1960, 1250)
                self.field_configs['school_code']['origin'] = (2120, 1250)
                self._current_answer_blocks = self.ANSWER_BLOCKS
                self._current_answer_row_positions = self.ANSWER_ROW_POSITIONS
            else:
                # Default 2025-26 Khoj batch - use FIELD_CONFIGS defaults
                # The defaults are calibrated from 41_626.jpg which works correctly:
                # name origin: (243, 690), roll: (1600, 1278), etc.
                # Don't overwrite - just use the values already set from FIELD_CONFIGS
                self._current_answer_blocks = self.ANSWER_BLOCKS
                self._current_answer_row_positions = self.ANSWER_ROW_POSITIONS

            # Extract fields using grid-based sampling
            if self.model:
                # Use YOLO model for bubble detection
                result = self._process_with_yolo(aligned)
            else:
                # Use direct grid sampling (more reliable)
                result = self._process_with_grid_sampling(aligned)

            # Always verify name with OCR and pick best result
            result = self._verify_name_with_ocr(result, image_path)

            return result

        except Exception as e:
            import traceback
            return OMRResult(
                success=False,
                error_message=f"Processing error: {str(e)}\n{traceback.format_exc()}"
            )

    def _process_with_new_pipeline(self, image: np.ndarray, image_path: str) -> OMRResult:
        """
        Process image using the new modular pipeline.

        Uses:
        1. MarkerAligner for perspective correction
        2. GridDetector for dynamic grid detection
        3. AdaptiveBubbleClassifier for ML-based classification
        4. FieldExtractor for value extraction

        This pipeline handles varying image dimensions and grid positions automatically.
        """
        try:
            # Step 1: Marker-based alignment
            marker_aligner = MarkerAligner()
            alignment_result = marker_aligner.align(image)

            if alignment_result.aligned_image is None:
                return OMRResult(
                    success=False,
                    error_message="Failed to align image"
                )

            aligned = alignment_result.aligned_image

            # Convert to grayscale for processing
            if len(aligned.shape) == 3:
                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
            else:
                gray = aligned

            # Step 2: Create extractor with adaptive classifier
            bubble_classifier = create_adaptive_classifier()
            grid_detector = GridDetector()
            field_extractor = FieldExtractor(bubble_classifier, grid_detector)

            # Step 3: Extract all fields (pass color image for batch type detection)
            extraction_result = field_extractor.extract_all_fields(gray, aligned)

            # Step 4: Convert to OMRResult
            result = OMRResult(
                success=True,
                name=extraction_result.name,
                roll_number=extraction_result.roll_number,
                class_=extraction_result.class_,
                section=extraction_result.section,
                school_code=extraction_result.school_code,
                answers=extraction_result.answers,
                confidence=extraction_result.overall_confidence,
                field_confidences={
                    'name': extraction_result.name_confidence,
                    'roll_number': extraction_result.roll_confidence,
                    'class': extraction_result.class_confidence,
                    'section': extraction_result.section_confidence,
                    'school_code': extraction_result.school_code_confidence,
                    'answers': extraction_result.answers_confidence,
                    'alignment_success': alignment_result.success,
                }
            )

            return result

        except Exception as e:
            import traceback
            # Fall back to legacy pipeline on error
            print(f"New pipeline error, falling back to legacy: {e}")
            return self._process_legacy(image, image_path)

    def _process_legacy(self, image: np.ndarray, image_path: str) -> OMRResult:
        """Legacy processing pipeline (fallback)."""
        # Align image to target dimensions
        aligned = self._align_image(image)

        # Auto-detect batch type
        self._is_chayan_batch = self._detect_best_origin(aligned)
        self._is_2023_form = False

        if self._is_chayan_batch:
            self.field_configs['name']['origin'] = self.field_configs['name'].get('origin_chayan', (180, 710))
            self._current_answer_blocks = self.ANSWER_BLOCKS_CHAYAN
            self._current_answer_row_positions = self.ANSWER_ROW_POSITIONS_CHAYAN
        else:
            self._current_answer_blocks = self.ANSWER_BLOCKS
            self._current_answer_row_positions = self.ANSWER_ROW_POSITIONS

        # Use grid sampling
        result = self._process_with_grid_sampling(aligned)
        return result

    def _verify_name_with_ocr(self, result: OMRResult, image_path: str) -> OMRResult:
        """Always run OCR verification and compare with bubble detection result."""
        if self.ocr_extractor is None:
            return result

        bubble_name = result.name or ""
        bubble_conf = result.field_confidences.get('name', 0.0)

        try:
            # ALWAYS run OCR extraction
            ocr_name, ocr_conf = self.ocr_extractor.extract_name(image_path)
            ocr_clean = self._clean_ocr_name(ocr_name) if ocr_name else ""

            # Compare and pick best result
            final_name, source = self._compare_name_results(
                bubble_name,
                ocr_clean,
                bubble_conf,
                ocr_conf
            )

            result.name = final_name
            result.ocr_used = (source == 'ocr')
            result.field_confidences['name_source'] = source
            result.field_confidences['ocr_confidence'] = ocr_conf
            result.field_confidences['bubble_name'] = bubble_name
            result.field_confidences['ocr_name'] = ocr_clean

        except Exception as e:
            # OCR failed, keep bubble result if valid
            if not bubble_name:
                result.error_message = f"OCR fallback failed: {e}"

        return result

    def _compare_name_results(
        self,
        bubble_name: str,
        ocr_name: str,
        bubble_conf: float,
        ocr_conf: float
    ) -> tuple:
        """
        Compare bubble detection and OCR results to pick the best name.

        Decision logic:
        1. Both blank -> return blank with low confidence
        2. One blank -> return non-blank result
        3. Both have content -> compare quality metrics
        """
        from typing import Tuple

        bubble_score = self._score_name_quality(bubble_name, bubble_conf)
        ocr_score = self._score_name_quality(ocr_name, ocr_conf)

        # Case 1: Both blank
        if not bubble_name and not ocr_name:
            return ("", "none")

        # Case 2: Only bubble has content
        if bubble_name and not ocr_name:
            return (bubble_name, "bubble")

        # Case 3: Only OCR has content
        if not bubble_name and ocr_name:
            return (ocr_name, "ocr")

        # Case 4: Both have content - use quality scoring
        # Strong bubble result (high score, valid pattern)
        if bubble_score > 0.7 and not self._is_suspicious_name(bubble_name):
            # Only prefer OCR if it's significantly better
            if ocr_score > bubble_score + 0.2 and ' ' in ocr_name:
                return (ocr_name, "ocr")
            return (bubble_name, "bubble")

        # Weak bubble result - prefer OCR if reasonable
        if ocr_score > 0.4 and ' ' in ocr_name and not self._is_suspicious_name(ocr_name):
            return (ocr_name, "ocr")

        # Default to bubble if both are weak but bubble has content
        return (bubble_name if bubble_name else ocr_name,
                "bubble" if bubble_name else "ocr")

    def _score_name_quality(self, name: str, confidence: float) -> float:
        """Score a name result based on quality indicators."""
        if not name:
            return 0.0

        score = confidence * 0.4  # Base from detection confidence

        # Length bonus (reasonable Indian names: 8-25 chars)
        name_len = len(name.replace(' ', ''))
        if 8 <= name_len <= 25:
            score += 0.2
        elif 5 <= name_len < 8:
            score += 0.1

        # Space bonus (most Indian names have 2+ words)
        if ' ' in name:
            word_count = len(name.split())
            if word_count >= 2:
                score += 0.2
            if word_count >= 3:
                score += 0.1

        # Penalty for suspicious patterns
        if self._is_suspicious_name(name):
            score -= 0.3

        return min(1.0, max(0.0, score))

    def _is_suspicious_name(self, name: str) -> bool:
        """Check if a name looks suspicious (likely detection error)."""
        if not name or len(name) < 3:
            return True

        words = name.upper().split()
        vowels = 'AEIOU'

        # Common Indian name patterns - if NO word matches, it's suspicious
        # First names often end in: A, I, AN, AR, AT, AV, AY, EE, EK, EN, ER, ESH, EV,
        # IK, IL, IN, IR, ISH, IT, OJ, UL, UM, UN, UR, USH
        common_endings = ['AR', 'AN', 'AV', 'AY', 'AT', 'AJ', 'AK', 'AL', 'AM',
                         'EE', 'EK', 'EN', 'ER', 'EV', 'ESH',
                         'IK', 'IL', 'IN', 'IR', 'IT', 'ISH', 'IV',
                         'OJ', 'OK', 'OM',
                         'UL', 'UM', 'UN', 'UR', 'USH', 'UV',
                         'TA', 'TI', 'NA', 'NI', 'RI', 'YA', 'HI', 'SH', 'TH']

        # Common starts for Indian names
        common_starts = ['A', 'B', 'C', 'D', 'G', 'H', 'J', 'K', 'L', 'M', 'N',
                        'P', 'R', 'S', 'T', 'U', 'V', 'Y',
                        'SH', 'CH', 'TH', 'KR', 'PR', 'SR', 'DR', 'BR', 'GR', 'TR',
                        'AN', 'AR', 'AS', 'AK', 'AM', 'AD', 'AB', 'AV', 'AY',
                        'RA', 'RO', 'RI', 'RU', 'RE',
                        'KA', 'KU', 'KI', 'KO', 'KE',
                        'SA', 'SU', 'SI', 'SO', 'SE',
                        'MA', 'MU', 'MI', 'MO', 'ME',
                        'PA', 'PU', 'PI', 'PO', 'PE',
                        'NA', 'NU', 'NI', 'NO', 'NE',
                        'VI', 'VA', 'VE', 'VO', 'VU']

        # Words that are definitely NOT names
        not_names = ['OIL', 'GAS', 'THE', 'AND', 'FOR', 'BUT', 'NOT', 'YOU', 'ALL',
                    'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS',
                    'HIM', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'WAY',
                    'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO',
                    'USE', 'AGU', 'MUA', 'CULL', 'APRFARNY', 'NT', 'SK', 'RU',
                    'GMU', 'QBH', 'MCF', 'FBNE', 'HDTLLAANLOE', 'TRTMAALOT']

        for word in words:
            if len(word) < 2:
                continue

            # Check if word is in not-names list
            if word in not_names:
                return True

            # Check vowel ratio - real names have 25-60% vowels
            vowel_count = sum(1 for c in word if c in vowels)
            vowel_ratio = vowel_count / len(word) if len(word) > 0 else 0

            # Too few vowels is suspicious
            if len(word) > 4 and vowel_ratio < 0.20:
                return True

            # Check for too many repeated consonants
            repeat_count = 0
            for i in range(len(word) - 1):
                if word[i] == word[i+1]:
                    repeat_count += 1
            if repeat_count >= 2:  # Two or more doubled letters
                return True

            # Check for suspicious 2-letter combos that don't occur in Indian names
            bad_combos = ['RD', 'TL', 'DF', 'QR', 'XY', 'ZK', 'ZB', 'ZD', 'ZF', 'ZG',
                         'FN', 'FD', 'FB', 'LR', 'RL', 'NR', 'RN', 'WR', 'WN', 'WL']
            for i in range(len(word) - 1):
                if word[i:i+2] in bad_combos:
                    return True

        # Check for extremely long words (Indian names can be long, e.g., CHANDRASHEKHAR VISHWAKARMA = 25)
        name_no_space = name.replace(' ', '')
        if len(name_no_space) > 28:
            return True

        # Check for alternating patterns (ABABAB)
        if len(name_no_space) >= 8:
            pattern_count = 0
            for i in range(len(name_no_space) - 3):
                if name_no_space[i] == name_no_space[i+2] and name_no_space[i+1] == name_no_space[i+3]:
                    pattern_count += 1
            if pattern_count >= 2:
                return True

        return False

    def _clean_ocr_name(self, ocr_name: str) -> str:
        """Clean OCR result to extract valid name."""
        import re

        if not ocr_name:
            return ""

        # Remove common OCR artifacts
        name = ocr_name.upper()
        name = re.sub(r'[^A-Z\s]', '', name)  # Keep only letters and spaces
        name = ' '.join(name.split())  # Normalize spaces

        # Try to extract surname from OCR result
        words = name.split()
        if len(words) < 2:
            return name

        # Keep words that look like names (length >= 3)
        valid_words = [w for w in words if len(w) >= 3]

        if len(valid_words) >= 2:
            return ' '.join(valid_words[:3])  # Max 3 words
        elif len(valid_words) == 1:
            return valid_words[0]
        else:
            return name

    def _find_corner_markers(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Find the 4 corner registration markers on the OMR sheet.

        These markers are used for perspective correction to align all sheets
        to a consistent reference frame, regardless of scan quality or year.

        Uses expected corner positions to filter candidates and avoid picking
        wrong dark regions on the form.

        Returns:
            4x2 numpy array of corner coordinates [TL, TR, BL, BR], or None if not found
        """
        h, w = gray.shape

        # Expected corner positions (approximate, allow tolerance)
        # These are the expected marker positions before alignment
        EXPECTED_CORNERS = {
            'tl': (170, 530),     # Top-left
            'tr': (w - 170, 530), # Top-right
            'bl': (170, 3150),    # Bottom-left
            'br': (w - 170, 3150) # Bottom-right
        }
        # Maximum distance from expected position to accept a candidate
        MAX_DISTANCE = 200

        # Threshold to find dark regions (markers are solid black)
        _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for marker-like contours (solid, roughly square, right size)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 500 < area < 2500:  # Marker size range
                x, y, cw, ch = cv2.boundingRect(contour)
                aspect = cw / ch if ch > 0 else 0
                if 0.5 < aspect < 2.0:  # Roughly square
                    roi = gray[y:y+ch, x:x+cw]
                    mean_val = np.mean(roi)
                    if mean_val < 80:  # Dark enough to be a marker
                        cx, cy = x + cw//2, y + ch//2
                        candidates.append((cx, cy, area, mean_val))

        if len(candidates) < 4:
            return None

        def find_best_corner(corner_name, expected_pos, candidates):
            """Find the best candidate for a corner, preferring ones close to expected position."""
            exp_x, exp_y = expected_pos
            # Filter to candidates within MAX_DISTANCE of expected position
            nearby = []
            for c in candidates:
                dist = np.sqrt((c[0] - exp_x)**2 + (c[1] - exp_y)**2)
                if dist < MAX_DISTANCE:
                    nearby.append((c, dist))

            if not nearby:
                # Fallback: use all candidates in the correct quadrant
                return None

            # Sort by distance to expected position
            nearby.sort(key=lambda x: x[1])
            return nearby[0][0]  # Return best candidate

        # Find corners using expected positions
        tl = find_best_corner('tl', EXPECTED_CORNERS['tl'], candidates)
        tr = find_best_corner('tr', EXPECTED_CORNERS['tr'], candidates)
        bl = find_best_corner('bl', EXPECTED_CORNERS['bl'], candidates)
        br = find_best_corner('br', EXPECTED_CORNERS['br'], candidates)

        # Fallback to quadrant-based selection if expected position matching fails
        if not all([tl, tr, bl, br]):
            mid_x, mid_y = w // 2, h // 2

            tl_candidates = [c for c in candidates if c[0] < mid_x and c[1] < mid_y]
            tr_candidates = [c for c in candidates if c[0] > mid_x and c[1] < mid_y]
            bl_candidates = [c for c in candidates if c[0] < mid_x and c[1] > mid_y]
            br_candidates = [c for c in candidates if c[0] > mid_x and c[1] > mid_y]

            if not all([tl_candidates, tr_candidates, bl_candidates, br_candidates]):
                return None

            if tl is None:
                tl = min(tl_candidates, key=lambda c: c[0] + c[1])
            if tr is None:
                tr = max(tr_candidates, key=lambda c: c[0] - c[1])
            if bl is None:
                bl = min(bl_candidates, key=lambda c: c[0] - c[1])
            if br is None:
                br = max(br_candidates, key=lambda c: c[0] + c[1])

        return np.float32([
            [tl[0], tl[1]],  # Top-left
            [tr[0], tr[1]],  # Top-right
            [bl[0], bl[1]],  # Bottom-left
            [br[0], br[1]],  # Bottom-right
        ])

    def _align_image(self, image: np.ndarray) -> np.ndarray:
        """Align image using simple resize.

        Stores original dimensions for offset calculation.
        """
        h, w = image.shape[:2]

        # Store original size for offset calculation
        self._original_size = (w, h)

        # Simple resize to target dimensions
        if w == self.TARGET_WIDTH and h == self.TARGET_HEIGHT:
            resized = image
        else:
            resized = cv2.resize(image, (self.TARGET_WIDTH, self.TARGET_HEIGHT))

        self._markers_aligned = False
        return resized

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image brightness/contrast for consistent bubble detection.

        Different scanners produce images with varying brightness levels.
        This normalization ensures consistent detection thresholds.
        """
        # Convert to LAB color space for better brightness normalization
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # This normalizes contrast while preserving local details
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_norm = clahe.apply(l)

            # Merge back
            lab_norm = cv2.merge([l_norm, a, b])
            normalized = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            normalized = clahe.apply(image)

        return normalized

    def _align_with_reference(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Align image using feature matching with reference."""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            if len(self.reference_image.shape) == 3:
                ref_gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
            else:
                ref_gray = self.reference_image.copy()

            # Detect ORB features
            orb = cv2.ORB_create(2000)
            kp1, des1 = orb.detectAndCompute(gray, None)
            kp2, des2 = orb.detectAndCompute(ref_gray, None)

            if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
                return None

            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            # Use top matches
            good_matches = matches[:int(len(matches) * 0.2)]

            if len(good_matches) < 4:
                return None

            # Extract matched points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find homography
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                return None

            # Warp image to reference dimensions
            h, w = ref_gray.shape
            return cv2.warpPerspective(image, H, (w, h))

        except Exception:
            return None

    def _calibrate_name_grid(self, gray: np.ndarray) -> dict:
        """Auto-calibrate name grid parameters using circle detection.

        Returns dict with: origin, rows_gap, cols_gap, x_shift_per_row
        Returns None if calibration fails.

        This function tries multiple detection thresholds to handle varying
        image quality/contrast across different scans.
        """
        # Use HoughCircles to detect bubbles in the name region
        roi_x1, roi_y1 = 130, 600
        roi_x2, roi_y2 = 900, 2100

        if roi_y2 > gray.shape[0] or roi_x2 > gray.shape[1]:
            return None

        roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]

        # Try multiple param2 thresholds from strict to lenient
        # Different scans have different contrast levels
        for param2 in [30, 25, 20, 15]:
            circles = cv2.HoughCircles(
                roi, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                param1=50, param2=param2, minRadius=10, maxRadius=20
            )

            if circles is None or len(circles[0]) < 50:
                continue

            # Convert to absolute coordinates
            all_circles = []
            for circle in circles[0]:
                x, y, r = circle
                all_circles.append((int(x + roi_x1), int(y + roi_y1), int(r)))

            # Sort by Y to find rows
            sorted_by_y = sorted(all_circles, key=lambda c: c[1])

            # Group into rows (circles with similar Y values)
            rows = []
            current_row = [sorted_by_y[0]]
            for circle in sorted_by_y[1:]:
                if abs(circle[1] - current_row[-1][1]) < 25:
                    current_row.append(circle)
                else:
                    # Filter out rows with too many circles (noise detection)
                    # Name grid has max ~12-15 circles per row
                    if 5 <= len(current_row) <= 20:
                        rows.append(sorted(current_row, key=lambda c: c[0]))
                    current_row = [circle]
            if 5 <= len(current_row) <= 20:
                rows.append(sorted(current_row, key=lambda c: c[0]))

            if len(rows) < 5:
                continue

            # Calculate row spacing (Y gap)
            row_y_positions = [np.mean([c[1] for c in row]) for row in rows]
            row_gaps = [row_y_positions[i+1] - row_y_positions[i] for i in range(len(row_y_positions)-1)]
            avg_row_gap = int(np.median(row_gaps))

            # Validate row gap is reasonable (should be ~50-55 pixels)
            if not (40 <= avg_row_gap <= 65):
                continue

            # Calculate column spacing (X gap) from a consistent row
            # Find a row with typical number of circles (10-13)
            good_row = None
            for row in rows:
                if 8 <= len(row) <= 15:
                    good_row = row
                    break
            if good_row is None:
                good_row = rows[0]

            if len(good_row) < 3:
                continue
            col_gaps = [good_row[i+1][0] - good_row[i][0] for i in range(len(good_row)-1)]
            avg_col_gap = int(np.median(col_gaps))

            # Validate col gap is reasonable (should be ~50-55 pixels)
            if not (40 <= avg_col_gap <= 65):
                continue

            # Calculate grid skew angle (how much X shifts per row)
            first_col_x = [row[0][0] for row in rows[:min(10, len(rows))]]

            if len(first_col_x) >= 2:
                x_shift_per_row = (first_col_x[-1] - first_col_x[0]) / (len(first_col_x) - 1)
            else:
                x_shift_per_row = 0

            # Find origin (first bubble of first row)
            origin_x = rows[0][0][0]
            origin_y = rows[0][0][1]

            # Validate origin is in expected range for name grid
            # Based on working image 41_626.jpg: origin=(243, 690)
            # Name grid origin should be roughly at X: 220-280, Y: 670-720
            if not (220 <= origin_x <= 280 and 670 <= origin_y <= 720):
                continue

            # Validate x_shift is in expected range (grid skews left, so negative)
            # Should be between -8 and +2 typically
            if not (-10 <= x_shift_per_row <= 3):
                continue

            return {
                'origin': (origin_x, origin_y),
                'rows_gap': avg_row_gap,
                'cols_gap': avg_col_gap,
                'x_shift_per_row': x_shift_per_row
            }

        # All thresholds failed
        return None

    def _detect_name_grid_origin(self, gray: np.ndarray) -> tuple:
        """Auto-detect the name grid origin by finding bubble positions."""
        # Use HoughCircles to detect bubbles in the name region
        # Search a wider area to find proper grid rows
        name_region_y1, name_region_y2 = 650, 1100  # Include more rows
        name_region_x1, name_region_x2 = 100, 800   # Include more columns

        if name_region_y2 > gray.shape[0] or name_region_x2 > gray.shape[1]:
            return None

        region = gray[name_region_y1:name_region_y2, name_region_x1:name_region_x2]

        circles = cv2.HoughCircles(
            region,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=12,
            maxRadius=20
        )

        if circles is None or len(circles[0]) < 10:
            return None

        # Adjust coordinates to full image
        circles = circles[0]
        adjusted = [(int(c[0] + name_region_x1), int(c[1] + name_region_y1), int(c[2])) for c in circles]

        # Group bubbles into rows by y-coordinate
        adjusted.sort(key=lambda c: c[1])
        rows = []
        current_row = [adjusted[0]]

        for c in adjusted[1:]:
            if abs(c[1] - current_row[0][1]) < 25:  # Same row
                current_row.append(c)
            else:
                if len(current_row) >= 8:  # Valid row has at least 8 bubbles
                    rows.append(sorted(current_row, key=lambda x: x[0]))
                current_row = [c]
        if len(current_row) >= 8:
            rows.append(sorted(current_row, key=lambda x: x[0]))

        # Find the topmost row that has:
        # 1. First bubble x position around 175-210 (column 0)
        # 2. Consistent column gap of 50-60 pixels
        # 3. At least 10 bubbles in the row
        for row in rows:
            if len(row) < 10:
                continue
            first_bubble = row[0]
            # Check column gap consistency
            if len(row) >= 3:
                gap1 = row[1][0] - row[0][0]
                gap2 = row[2][0] - row[1][0]
                avg_gap = (gap1 + gap2) / 2
                # Valid bubble grid has consistent ~54 pixel gap
                if 48 < avg_gap < 60 and abs(gap1 - gap2) < 10:
                    # First column should start around x=175-210
                    if 165 < first_bubble[0] < 220:
                        # This is likely the actual bubble grid row A
                        origin_x = first_bubble[0] - first_bubble[2]
                        origin_y = first_bubble[1] - first_bubble[2]
                        return (origin_x, origin_y)

        return None

    def _process_with_grid_sampling(self, image: np.ndarray) -> OMRResult:
        """Process OMR sheet using simplified corridor-based detection.

        Uses unified detection logic for all fields:
        - Loop through all bubbles in grid
        - Check each bubble using corridor point sampling
        - First filled bubble wins (no complex gap analysis)
        """
        result = OMRResult(success=True)

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Auto-calibrate name grid parameters per image using circle detection
        # This handles varying alignment/skew between different scanned images
        name_calib = self._calibrate_name_grid(gray)
        if name_calib:
            self.field_configs['name']['origin'] = name_calib['origin']
            self.field_configs['name']['rows_gap'] = name_calib['rows_gap']
            self.field_configs['name']['cols_gap'] = name_calib['cols_gap']
            self.field_configs['name']['x_shift_per_row'] = name_calib['x_shift_per_row']
            import sys
            print(f"[DEBUG] Name grid auto-calibrated: origin={name_calib['origin']}, "
                  f"x_shift={name_calib['x_shift_per_row']:.2f}", file=sys.stderr)

        # Extract all fields using pure ML detection (if available) or threshold fallback
        # Pure ML is more consistent and handles edge cases better
        if self.bubble_classifier:
            # Name: 25 columns × 26 rows (A-Z), returns 25-char string
            result.name = self._sample_field_ml(gray, 'name')
            result.field_confidences['name'] = 1.0
            result.ocr_used = False

            # Roll number: 3 columns × 10 rows (0-9)
            result.roll_number = self._sample_field_ml(gray, 'roll_number').strip()
            result.field_confidences['roll_number'] = 1.0

            # Class: 1 column × 8 rows (3-10)
            result.class_ = self._sample_field_ml(gray, 'class').strip()
            result.field_confidences['class'] = 1.0

            # Section: 1 column × 8 rows (A-H)
            result.section = self._sample_field_ml(gray, 'section').strip()
            result.field_confidences['section'] = 1.0

            # School code: 3 columns × 10 rows (0-9)
            result.school_code = self._sample_field_ml(gray, 'school_code').strip()
            result.field_confidences['school_code'] = 1.0
        else:
            # Fallback to threshold-based detection
            result.name = self._detect_field_simple(gray, 'name')
            result.field_confidences['name'] = 1.0
            result.ocr_used = False

            result.roll_number = self._sample_field_threshold(gray, 'roll_number').strip()
            result.field_confidences['roll_number'] = 1.0

            result.class_ = self._sample_field_threshold(gray, 'class').strip()
            result.field_confidences['class'] = 1.0

            result.section = self._sample_field_threshold(gray, 'section').strip()
            result.field_confidences['section'] = 1.0

            result.school_code = self._sample_field_threshold(gray, 'school_code').strip()
            result.field_confidences['school_code'] = 1.0

        # Answers: 60 questions × 4 options (A-D)
        # Use dynamic detection which finds actual bubble positions using HoughCircles
        # This is more robust to alignment variations than fixed coordinates
        result.answers = self._detect_answers_dynamic(gray)
        result.field_confidences['answers'] = 1.0

        # Set overall confidence (simplified - all fields use same detection logic)
        result.confidence = 100

        return result

    def _clean_detected_name(self, raw_name: str) -> str:
        """Clean up detected name by removing spurious characters and adding spaces."""
        import re

        if not raw_name:
            return raw_name

        name = raw_name.strip().upper()

        # Comprehensive list of Indian surnames
        surnames_by_len = [
            'VISHWAKARMA', 'SRIVASTAVA', 'CHATTERJEE', 'MUKHERJEE', 'BANERJEE', 'PRAJAPATI',
            'KERKETTA', 'HEMBRAM', 'RAJWAR', 'MAHATO', 'THAKUR', 'PASWAN', 'RAWANI', 'PRAKASH',
            'BHARTI', 'MANDAL', 'CHANDRA', 'KUMARI', 'PRASAD', 'SHARMA', 'MISHRA', 'MAURYA',
            'PANDEY', 'TIWARI', 'HANSDA', 'TIRKEY', 'KHATUN', 'PATHAK', 'ANSARI', 'KHATIK',
            'SHUKLA', 'DUBEY', 'VERMA', 'PANDIT', 'TOPPO', 'LAKRA', 'BASKI', 'SAHAY', 'ARYA',
            'KISKU', 'SOREN', 'ORAON', 'GUPTA', 'YADAV', 'KACHHAP', 'MARANDI', 'PRIYA', 'KAUR',
            'KUMAR', 'SINGH', 'MURMU', 'SHAW', 'TUDU', 'MODI', 'TURI', 'DEVI', 'RANI', 'KARN',
            'MUNDA', 'HORO', 'EKKA', 'MINJ', 'BECK', 'TIGGA', 'KUJUR', 'BARA', 'NATH', 'GOND',
            'DAS', 'PAL', 'LAL', 'RAM', 'SAH', 'ROY', 'RAJ',
        ]
        surnames_sorted = sorted(surnames_by_len, key=len, reverse=True)

        def clean_word(word):
            """Clean a single word of garbage characters."""
            w = word
            # Remove repeated ARTIFACT characters only (WWWW -> W, MMM -> M, BBB -> B)
            # Don't collapse other repeated letters (AAA might be legitimate)
            for c in 'WMB':
                w = re.sub(rf'{c}{{3,}}', c, w)
            # Remove trailing W, M, B (common artifacts)
            w = re.sub(r'[WMB]+$', '', w)
            # Remove trailing patterns like WK, MW, BM
            w = re.sub(r'[WMB][WMB]$', '', w)
            # Remove single trailing artifact after valid ending
            if len(w) > 3 and w[-1] in 'WMB' and w[-2] not in 'WMB':
                w = w[:-1]
            return w

        # If input already has spaces (from detected empty columns), preserve them
        # Just clean each word individually
        # IMPORTANT: Keep single letters as they may be initials (common in Indian names)
        if ' ' in name:
            words = name.split()
            cleaned_words = []
            for word in words:
                cleaned = clean_word(word)
                if cleaned:
                    # Keep all words including single letters (initials)
                    # But filter out common artifact letters if they appear alone
                    artifact_letters = {'W', 'M', 'B', 'Q'}  # Common detection artifacts
                    if len(cleaned) == 1 and cleaned in artifact_letters:
                        # Skip single artifact letters
                        continue
                    cleaned_words.append(cleaned)
            return ' '.join(cleaned_words)

        # No spaces detected - apply full surname extraction logic
        name_no_space = name

        # Step 1: Remove trailing garbage patterns
        name_no_space = clean_word(name_no_space)

        # Step 2: Remove garbage after surnames
        for surname in surnames_sorted:
            pattern = rf'({surname})([A-Z]{{2,6}})$'
            match = re.search(pattern, name_no_space)
            if match:
                garbage = match.group(2)
                full_word = surname + garbage
                is_valid_extension = any(full_word.endswith(s) and s != surname
                                        for s in surnames_sorted if len(s) > len(surname))
                if is_valid_extension:
                    continue
                if (any(c in garbage for c in 'MW') or len(garbage) <= 3 or
                    re.search(r'(.)\1', garbage)):
                    name_no_space = name_no_space[:match.start()] + surname
                    break

        # Step 3: Find and extract surnames
        found_surnames = []
        remaining = name_no_space

        for iteration in range(3):
            found_one = False
            for surname in surnames_sorted:
                if remaining.endswith(surname):
                    potential_remaining = remaining[:-len(surname)]
                    potential_remaining_clean = re.sub(r'[MW]+$', '', potential_remaining)
                    if len(surname) <= 4 and len(potential_remaining_clean) <= 3:
                        continue
                    found_surnames.insert(0, surname)
                    remaining = potential_remaining
                    remaining = re.sub(r'[MW]{2,}$', '', remaining)
                    if remaining.endswith('W'):
                        remaining = remaining[:-1]
                    if remaining.endswith('M') and len(remaining) > 4:
                        if not (remaining.endswith('AM') or remaining.endswith('UM') or remaining.endswith('IM')):
                            remaining = remaining[:-1]
                    found_one = True
                    break
            if not found_one and iteration > 0:
                for surname in surnames_sorted:
                    idx = remaining.rfind(surname)
                    if idx > 0:
                        after = remaining[idx + len(surname):]
                        if 0 < len(after) <= 5 and re.match(r'^[A-Z]*$', after):
                            found_surnames.insert(0, surname)
                            remaining = remaining[:idx]
                            found_one = True
                            break
            if not found_one:
                break

        # Step 4: Clean first name
        first_name = remaining
        first_name = re.sub(r'[MW]{2,}$', '', first_name)
        if first_name.endswith('W'):
            first_name = first_name[:-1]
        if len(first_name) > 4 and first_name.endswith('AA'):
            first_name = first_name[:-1]

        # Step 5: Build final name
        parts = []
        # Keep first name if it's at least 2 chars, OR if it's all we have (no surnames)
        if first_name and (len(first_name) >= 2 or not found_surnames):
            parts.append(first_name)
        parts.extend(found_surnames)
        name = ' '.join(parts)

        # Step 6: Clean repeated chars
        name = re.sub(r'M{2,}', 'M', name)
        name = re.sub(r'W{2,}', 'W', name)
        name = re.sub(r'([A-Z])\1{3,}', r'\1', name)
        name = re.sub(r'X{2,}[A-Z]*', '', name)
        name = ' '.join(name.split())

        return name

    def _detect_best_origin(self, aligned_bgr: np.ndarray) -> bool:
        """
        Auto-detect whether to use Chayan or Khoj origin by form color.

        Chayan forms are BLUE, Khoj forms are RED.

        Args:
            aligned_bgr: Aligned BGR color image

        Returns:
            True if Chayan batch (blue form), False if Khoj batch (red form)
        """
        # Sample from bubble grid areas where form color is visible
        # Multiple regions to get better average
        regions = [
            aligned_bgr[700:900, 200:400],    # Name field area
            aligned_bgr[1500:1700, 1200:1400],  # Center bubble area
            aligned_bgr[2200:2400, 500:700],    # Answer grid area
        ]

        # Calculate average color across all regions
        all_blues = []
        all_greens = []
        all_reds = []

        for region in regions:
            if region.size > 0:
                all_blues.append(np.mean(region[:, :, 0]))
                all_greens.append(np.mean(region[:, :, 1]))
                all_reds.append(np.mean(region[:, :, 2]))

        avg_b = np.mean(all_blues) if all_blues else 255
        avg_g = np.mean(all_greens) if all_greens else 255
        avg_r = np.mean(all_reds) if all_reds else 255

        # Detect dominant color
        # Blue forms: B > R (blue dominant)
        # Red forms: R > B (red dominant)
        is_blue = avg_b > avg_r + 5  # Blue > Red by at least 5

        # Debug output
        import sys
        print(f"[DEBUG] Form color BGR: B={avg_b:.1f}, G={avg_g:.1f}, R={avg_r:.1f}", file=sys.stderr)
        print(f"[DEBUG] Detected batch: {'Chayan (BLUE)' if is_blue else 'Khoj (RED)'}", file=sys.stderr)

        return is_blue

    def _detect_form_year(self, aligned_bgr: np.ndarray) -> bool:
        """
        Detect form year (2023-24 vs 2025-26) based on layout differences.

        Uses ML classifier if available, otherwise falls back to threshold detection.

        Returns:
            True if 2023-24 form, False if 2025-26 form
        """
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)

        # Test both origins
        # 2025-26: Calibrated from circle detection (2026-01-02)
        # 2023-24: Legacy coordinates
        origins_to_test = [
            ('2025-26', 243, 690, 51, -3.56),  # origin_x, origin_y, rows_gap, x_shift_per_row
            ('2023-24', 220, 610, 54, 0),       # Legacy layout (no skew)
        ]

        bubble_size = (33, 33)
        cols_gap = 52  # Updated from calibration
        bw, bh = bubble_size

        best_version = '2025-26'
        best_score = 0

        for version, origin_x, origin_y, rows_gap, x_shift_per_row in origins_to_test:
            filled_count = 0
            total_gap = 0

            # Test first 10 columns
            for col in range(10):
                base_x = origin_x + col * cols_gap
                bubble_data = []

                for row in range(26):
                    # Apply skew correction: X shifts as we go down rows
                    x = int(base_x + row * x_shift_per_row)
                    y = origin_y + row * rows_gap
                    if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                        bubble = gray[y:y+bh, x:x+bw].copy()
                        darkness = np.mean(bubble)
                        bubble_data.append((row, bubble, darkness))

                if not bubble_data:
                    continue

                # Sort by darkness
                sorted_data = sorted(bubble_data, key=lambda x: x[2])
                darkest_row, darkest_bubble, darkest_val = sorted_data[0]
                second_val = sorted_data[1][2] if len(sorted_data) > 1 else 255
                gap = second_val - darkest_val

                # Use ML classifier if available
                if self.bubble_classifier:
                    bubble_resized = cv2.resize(darkest_bubble, (28, 28))
                    predictions = self.bubble_classifier.predict_batch([bubble_resized])
                    is_filled, conf = predictions[0]
                    if is_filled and conf > 0.8:
                        filled_count += 1
                        total_gap += gap
                else:
                    # Fallback to threshold
                    if gap > 30 and darkest_val < 190:
                        filled_count += 1
                        total_gap += gap

            # Score based on filled bubbles and gap quality
            # More filled bubbles with higher gaps = better alignment
            score = filled_count * 100 + total_gap

            import sys
            print(f"[DEBUG] Form year test {version}: filled={filled_count}, gap={total_gap:.0f}, score={score:.0f}", file=sys.stderr)

            if score > best_score:
                best_score = score
                best_version = version

        is_2023 = (best_version == '2023-24')

        import sys
        print(f"[DEBUG] Selected form year: {best_version} (score={best_score:.0f})", file=sys.stderr)

        return is_2023

    def _calibrate_name_y_offset(self, gray: np.ndarray, origin: Tuple[int, int],
                                  bubble_size: Tuple[int, int], rows_gap: int,
                                  cols_gap: int, num_rows: int) -> int:
        """Find the best Y offset for name field alignment.

        Different scans can have vertical misalignment of 10-40 pixels.
        This method samples columns and finds the offset that gives
        the best bubble separation (largest gaps between filled and unfilled).

        Uses gap-based scoring with validation to reject garbage patterns.

        Returns:
            Best y_offset to apply (can be positive or negative)
        """
        # Test columns 0-9 (first 10 letters of name)
        test_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # With dynamic scaling from FIELD_CONFIGS_ORIGINAL, coordinates are already calibrated
        # Only need small adjustments (±20 pixels max) to handle minor scan variations
        # Using a tighter range prevents picking wildly wrong offsets
        y_offsets_to_test = list(range(-20, 21, 5))  # ±20 pixels for all cases

        # Collect scores for all offsets
        offset_scores = []

        for y_offset in y_offsets_to_test:
            gaps = []

            for col in test_cols:
                x = origin[0] + col * cols_gap

                # Sample all rows at this offset
                bubble_values = []
                for row in range(num_rows):
                    y = origin[1] + y_offset + row * rows_gap
                    bw, bh = bubble_size
                    if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                        bubble = gray[y:y+bh, x:x+bw]
                        darkness = np.mean(bubble)
                        bubble_values.append(darkness)

                if len(bubble_values) >= 2:
                    sorted_vals = sorted(bubble_values)
                    gap = sorted_vals[1] - sorted_vals[0]
                    gaps.append(gap)

            if len(gaps) >= 3:
                good_gaps = [g for g in gaps if g > 30]
                if good_gaps:
                    score = len(good_gaps) * np.mean(good_gaps)
                    offset_scores.append((y_offset, score))

        # Sort by score descending
        offset_scores.sort(key=lambda x: x[1], reverse=True)

        # Ensure offset 0 is tested (important baseline)
        zero_in_list = any(o == 0 for o, s in offset_scores[:10])
        if not zero_in_list:
            # Add offset 0 with a baseline score
            offset_scores.append((0, 50))

        # Try top candidates and validate detected pattern
        values = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        vowels = set('AEIOU')
        # Rare letter pairs in Indian names (suggest wrong alignment)
        garbage_patterns = {'QA', 'QS', 'QU', 'QO', 'QE', 'QI', 'LQ', 'ZQ', 'XQ',
                           'JQ', 'ZZ', 'QQ', 'XX', 'JJ', 'ZJ', 'JZ', 'XZ', 'ZX'}

        # Collect valid candidates with priority_first, then prefer offset closest to 0
        priority_candidates = []  # (y_offset, score, detected_letters)
        other_valid_candidates = []  # Valid but without priority first letter

        for y_offset, score in offset_scores[:10]:  # Try top 10 candidates
            # Quick detection at this offset
            detected_letters = []
            for col in range(8):  # Check first 8 columns
                x = origin[0] + col * cols_gap
                bubble_data = []
                for row in range(num_rows):
                    y = origin[1] + y_offset + row * rows_gap
                    bw, bh = bubble_size
                    if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                        bubble = gray[y:y+bh, x:x+bw]
                        darkness = np.mean(bubble)
                        bubble_data.append((row, darkness))

                if bubble_data:
                    median_val = np.median([v for _, v in bubble_data])
                    # Find ALL bubbles that qualify as filled, then pick FIRST one
                    filled_candidates = []
                    for row, darkness in bubble_data:
                        bubble_gap = median_val - darkness
                        if darkness < 180 and bubble_gap > 25:
                            filled_candidates.append((row, darkness))

                    if filled_candidates:
                        # Pick the FIRST filled (lowest row number)
                        filled_candidates.sort(key=lambda x: x[0])
                        best_row = filled_candidates[0][0]
                        detected_letters.append(values[best_row])

            # Validate detected pattern (need at least 2 letters for validation)
            if len(detected_letters) >= 2:
                detected_str = ''.join(detected_letters)
                # Check for garbage patterns
                has_garbage = any(p in detected_str for p in garbage_patterns)
                # Check for at least one vowel in first 6 letters
                has_vowel = any(c in vowels for c in detected_str[:6])
                # Reject offsets where first letter is unlikely (Q, W, X, Z, etc.)
                unlikely_first = set('QWXZFOU')  # Very rare as first letters
                first_letter_unlikely = detected_letters[0] in unlikely_first if detected_letters else True
                # Check for repeated same letter in first 4 positions (suspicious pattern)
                first_4 = detected_letters[:4]
                has_repeat = len(first_4) > len(set(first_4)) + 1  # Allow 1 repeat max
                # Very common starting letters get priority (A is most common for Indian names)
                priority_first = detected_letters[0] in 'ABCDKMNPRS' if detected_letters else False

                if not has_garbage and has_vowel and not first_letter_unlikely and not has_repeat:
                    if priority_first:
                        priority_candidates.append((y_offset, score, detected_letters))
                    else:
                        other_valid_candidates.append((y_offset, score, detected_letters))

        # From priority candidates, prefer the one closest to Y=0 for stability
        # This prevents wrong alignment when multiple offsets detect valid patterns
        if priority_candidates:
            # Sort by absolute offset value (prefer closer to 0)
            priority_candidates.sort(key=lambda x: abs(x[0]))
            return priority_candidates[0][0]

        # Second choice: valid candidates without priority first letter, prefer closest to 0
        if other_valid_candidates:
            other_valid_candidates.sort(key=lambda x: abs(x[0]))
            return other_valid_candidates[0][0]

        # Fallback to highest scoring offset
        return offset_scores[0][0] if offset_scores else 0

    def _calibrate_name_x_offset(self, gray: np.ndarray, origin: Tuple[int, int],
                                  bubble_size: Tuple[int, int], rows_gap: int,
                                  cols_gap: int, num_rows: int, y_offset: int) -> int:
        """Calibrate X offset for name field by finding alignment with best bubble detection.

        Different scans can have horizontal shifts of 10-30 pixels. This finds the
        x_offset that produces the clearest bubble detection (largest gaps).
        """
        # Chayan batch has pre-calibrated X origin, only need small adjustments
        if hasattr(self, '_is_chayan_batch') and self._is_chayan_batch:
            x_offsets_to_test = list(range(-10, 11, 5))  # ±10 pixels for Chayan
        else:
            x_offsets_to_test = list(range(-40, 20, 5))  # Full range for Khoj
        offset_scores = []

        # Test on first 5 columns
        test_cols = list(range(5))

        for x_offset in x_offsets_to_test:
            gaps = []

            for col in test_cols:
                x = origin[0] + col * cols_gap + x_offset

                bubble_values = []
                for row in range(num_rows):
                    y = origin[1] + y_offset + row * rows_gap
                    bw, bh = bubble_size
                    if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                        bubble = gray[y:y+bh, x:x+bw]
                        darkness = np.mean(bubble)
                        bubble_values.append(darkness)

                if len(bubble_values) >= 2:
                    sorted_vals = sorted(bubble_values)
                    gap = sorted_vals[1] - sorted_vals[0]
                    gaps.append(gap)

            if len(gaps) >= 3:
                good_gaps = [g for g in gaps if g > 30]
                if good_gaps:
                    score = len(good_gaps) * np.mean(good_gaps)
                    offset_scores.append((x_offset, score))

        # Sort by score descending
        offset_scores.sort(key=lambda x: x[1], reverse=True)

        return offset_scores[0][0] if offset_scores else 0

    def _check_corridor_fill(self, gray: np.ndarray, x: int, y: int, bubble_size: Tuple[int, int]) -> Tuple[bool, float]:
        """Check if a bubble is filled by examining the corridor between letter and bubble edge.

        The corridor is the area between the printed letter (center) and the bubble outline (edge).
        If this area is dark, the bubble is filled. If it's light, only the printed letter is visible.

        Args:
            gray: Grayscale image
            x: X coordinate of bubble top-left
            y: Y coordinate of bubble top-left
            bubble_size: (width, height) of bubble

        Returns:
            (is_filled, corridor_mean): Whether the bubble appears filled and the mean corridor darkness
        """
        bw, bh = bubble_size
        bubble = gray[y:y+bh, x:x+bw]

        # Center of bubble
        cx, cy = bw // 2, bh // 2

        # Sample points in the corridor - between letter center and bubble edge
        # These points should be in the area that gets filled when marking the bubble
        # but are NOT part of the printed letter itself
        check_points = [
            # Top corridor (above letter, below bubble edge)
            (6, cx), (7, cx - 2), (7, cx + 2), (6, cx - 4), (6, cx + 4),
            # Bottom corridor
            (bh - 7, cx), (bh - 8, cx - 2), (bh - 8, cx + 2), (bh - 7, cx - 4), (bh - 7, cx + 4),
            # Left corridor
            (cy, 6), (cy - 2, 7), (cy + 2, 7), (cy - 4, 6), (cy + 4, 6),
            # Right corridor
            (cy, bw - 7), (cy - 2, bw - 8), (cy + 2, bw - 8), (cy - 4, bw - 7), (cy + 4, bw - 7),
        ]

        corridor_values = []
        for py, px in check_points:
            if 0 <= py < bh and 0 <= px < bw:
                corridor_values.append(bubble[py, px])

        if not corridor_values:
            return False, 255.0

        corridor_mean = np.mean(corridor_values)
        very_dark_count = sum(1 for v in corridor_values if v < 100)
        dark_count = sum(1 for v in corridor_values if v < 150)
        very_dark_ratio = very_dark_count / len(corridor_values)
        dark_ratio = dark_count / len(corridor_values)

        # Check the entire bubble for very dark pixels
        bubble = gray[y:y+bh, x:x+bw]
        total_very_dark = np.sum(bubble < 50)

        # A bubble is filled if it has significantly more very dark pixels
        # than just the printed letter (which typically has ~100-130 pixels < 50)
        # OR if the corridor shows clear fill indicators
        # IMPORTANT: Must have at least some very dark pixels (>50) to be considered filled
        # This prevents false positives from moderately dark printed letters
        # Lowered threshold to 100 to catch lighter fills (was 150)
        has_minimum_dark = total_very_dark > 50
        is_filled = (
            total_very_dark > 100 or  # More dark pixels than typical printed letter (lowered from 150)
            (has_minimum_dark and corridor_mean < 170 and dark_ratio > 0.30) or  # Corridor shows fill
            (has_minimum_dark and corridor_mean < 180 and very_dark_ratio > 0.20)  # Very dark in corridor
        )

        return is_filled, corridor_mean

    # =========================================================================
    # SIMPLIFIED DETECTION METHODS
    # These methods use a unified corridor-based approach for all fields
    # =========================================================================

    def _is_bubble_filled_simple(self, gray: np.ndarray, x: int, y: int,
                                  bubble_size: Tuple[int, int]) -> bool:
        """Check if bubble is filled using ML classifier with threshold fallback.

        Args:
            gray: Grayscale image
            x: X coordinate of bubble top-left
            y: Y coordinate of bubble top-left
            bubble_size: (width, height) of bubble

        Returns:
            True if bubble appears filled, False otherwise
        """
        bw, bh = bubble_size

        # Bounds check
        if not (0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw):
            return False

        # Extract bubble image
        bubble = gray[y:y+bh, x:x+bw]

        # Use ML bubble classifier if available (now that coordinates are fixed)
        if self.bubble_classifier is not None:
            try:
                is_filled, confidence = self.bubble_classifier.predict(bubble)
                return is_filled
            except:
                pass  # Fall back to threshold if ML fails

        # Fallback: threshold-based detection
        very_dark_pixels = np.sum(bubble < 50)
        return very_dark_pixels > 200

    def _detect_field_simple(self, gray: np.ndarray, field_name: str) -> str:
        """Unified detection for any field using row-by-row checking.

        Uses dynamic coordinate scaling based on the actual image dimensions.
        Coordinates from FIELD_CONFIGS_ORIGINAL are scaled by:
            scaled_coord = original_coord * (TARGET_SIZE / original_image_size)

        Args:
            gray: Grayscale image (must be aligned to TARGET dimensions)
            field_name: Name of field to detect

        Returns:
            Detected value string
        """
        # Get original config for dynamic scaling
        orig_config = self.FIELD_CONFIGS_ORIGINAL.get(field_name)
        if not orig_config:
            # Fall back to pre-scaled config for name field
            config = self.field_configs.get(field_name)
            if not config:
                return ""
            origin = config['origin']
            bubble_size = config['bubble_size']
            rows_gap = config['rows_gap']
            cols_gap = config.get('cols_gap', rows_gap)
        else:
            # Dynamic scaling based on actual image dimensions
            REF_WIDTH = 1662  # Reference image width (41_625.jpg)
            REF_HEIGHT = 2320  # Reference image height
            if hasattr(self, '_original_size'):
                orig_w, orig_h = self._original_size
                scale_x = self.TARGET_WIDTH / orig_w
                scale_y = self.TARGET_HEIGHT / orig_h
                # Width-based X offset: narrower images need positive offset (move right)
                width_diff = REF_WIDTH - orig_w
                height_diff = REF_HEIGHT - orig_h
                x_offset = int(width_diff * scale_x * 2.0)  # 2x the scaled difference
                # Height-based Y offset: shorter images need negative offset (move up)
                y_offset_height = int(height_diff * scale_y * -1.5)
            else:
                # Default scale for reference image
                scale_x = self.TARGET_WIDTH / 1662
                scale_y = self.TARGET_HEIGHT / 2320
                x_offset = 0
                y_offset_height = 0

            # Scale all coordinates with width/height-based corrections
            origin = (
                int(orig_config['origin'][0] * scale_x) + x_offset,
                int(orig_config['origin'][1] * scale_y) + y_offset_height
            )
            bubble_size = (
                int(orig_config['bubble_size'][0] * scale_x),
                int(orig_config['bubble_size'][1] * scale_y)
            )
            rows_gap = int(orig_config['rows_gap'] * scale_y)
            cols_gap = int(orig_config.get('cols_gap', orig_config['rows_gap']) * scale_x)

        config = orig_config if orig_config else self.field_configs.get(field_name)
        num_rows = config['rows']
        num_cols = config['cols']
        values = config['values']

        # Get Y drift per column (compensates for grid skew in name field)
        y_drift_per_col = config.get('y_drift_per_col', 0)

        result = []

        for col in range(num_cols):
            # Calculate X position
            x = origin[0] + col * cols_gap

            # For name field: handle extra gaps between name sections
            if field_name == 'name':
                if hasattr(self, '_is_chayan_batch') and self._is_chayan_batch:
                    # Chayan has gaps at col 5 and col 10 (between first/middle/last name)
                    extra_gaps = config.get('extra_gaps_chayan', {})
                    for gap_col, extra_px in extra_gaps.items():
                        if col > gap_col:
                            x += extra_px
                elif col > 10:
                    # Khoj: add extra spacing after column 10
                    x += 54

            # Check ALL bubbles in this column
            all_bubbles = []
            filled_bubbles = []
            for row in range(num_rows):
                # Apply Y drift per column (grid skew compensation)
                y = origin[1] + col * y_drift_per_col + row * rows_gap
                y1, y2 = y, y + bubble_size[1]
                x1, x2 = x, x + bubble_size[0]

                if 0 <= y1 and y2 <= gray.shape[0] and 0 <= x1 and x2 <= gray.shape[1]:
                    bubble = gray[y1:y2, x1:x2]
                    mean_val = np.mean(bubble)
                    very_dark = np.sum(bubble < 50)

                    all_bubbles.append((row, mean_val, very_dark))

                    # Check if this bubble is filled using threshold
                    is_filled = (
                        (very_dark > 200 and mean_val < 150) or  # Strong fill
                        (very_dark > 150 and mean_val < 120) or  # Moderate fill
                        (very_dark > 300)  # Very strong absolute fill
                    )

                    if is_filled:
                        filled_bubbles.append((row, mean_val, very_dark))

            if filled_bubbles:
                # If 2 or more bubbles are filled, pick the FIRST one (lowest row number)
                filled_bubbles.sort(key=lambda x: x[0])  # Sort by row number ascending
                first_filled_row = filled_bubbles[0][0]
                result.append(values[first_filled_row])
            elif all_bubbles:
                # Fallback: if no bubble meets threshold, check if one is clearly darker
                all_bubbles.sort(key=lambda x: -x[2])  # Sort by vdark descending
                darkest = all_bubbles[0]
                second = all_bubbles[1] if len(all_bubbles) > 1 else (0, 255, 0)

                # Pick darkest if it has significantly more dark pixels than second
                if darkest[2] > 50 and darkest[2] > second[2] * 2:
                    result.append(values[darkest[0]])
                else:
                    result.append(' ')
            else:
                result.append(' ')

        return ''.join(result)

    def _find_best_x_offset(self, gray: np.ndarray, origin: Tuple[int, int],
                             bubble_size: Tuple[int, int], rows_gap: int,
                             num_rows: int, values: List[str]) -> int:
        """Find the best X offset for a field by testing offsets and finding filled bubbles.

        Different scans can have horizontal shifts of 20-40 pixels.
        This method tests offsets and returns the one where a bubble is detected.

        Returns:
            Best x_offset to apply (can be positive or negative)
        """
        x_offsets_to_test = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20]

        for x_offset in x_offsets_to_test:
            x = origin[0] + x_offset

            # Check if any bubble in this column is filled
            for row in range(num_rows):
                y = origin[1] + row * rows_gap

                if self._is_bubble_filled_simple(gray, x, y, bubble_size):
                    return x_offset  # Found a filled bubble at this offset

        # No filled bubble found at any offset, return 0
        return 0

    def _find_best_y_offset(self, gray: np.ndarray, origin: Tuple[int, int],
                             bubble_size: Tuple[int, int], rows_gap: int, cols_gap: int,
                             num_rows: int, num_cols: int) -> int:
        """Find the best Y offset for a field by testing offsets and finding filled bubbles.

        Different scans can have vertical shifts. This method tests Y offsets
        and returns the one where the most bubbles are detected.

        Returns:
            Best y_offset to apply (can be positive or negative)
        """
        y_offsets_to_test = [-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100]
        best_y_offset = 0
        best_filled_count = 0

        for y_offset in y_offsets_to_test:
            filled_count = 0
            for col in range(num_cols):
                x = origin[0] + col * cols_gap
                for row in range(num_rows):
                    y = origin[1] + y_offset + row * rows_gap
                    if self._is_bubble_filled_simple(gray, x, y, bubble_size):
                        filled_count += 1

            # We expect num_cols filled bubbles (one per column)
            # The offset with count closest to num_cols is best
            if filled_count >= num_cols and filled_count > best_filled_count:
                best_filled_count = filled_count
                best_y_offset = y_offset

        return best_y_offset

    def _detect_answers_simple(self, gray: np.ndarray) -> Dict[int, str]:
        """Detect all 60 answers using dynamically scaled coordinates.

        Uses ANSWER_BLOCKS_ORIGINAL (measured on 1662x2320 reference image)
        and scales to the actual image dimensions, similar to name field detection.

        Args:
            gray: Grayscale image (must be aligned to TARGET dimensions)

        Returns:
            Dict mapping question number (1-60) to answer (A/B/C/D or empty string)
        """
        answers = {}

        # Get original image dimensions for dynamic scaling
        if not hasattr(self, '_original_size'):
            # Fallback to old behavior if no original size
            answer_blocks = self._current_answer_blocks if hasattr(self, '_current_answer_blocks') else self.ANSWER_BLOCKS
            row_positions = self._current_answer_row_positions if hasattr(self, '_current_answer_row_positions') else self.ANSWER_ROW_POSITIONS
            option_gap = self.ANSWER_OPTION_GAP
            bubble_size = self.ANSWER_BUBBLE_SIZE
        else:
            # Dynamic scaling from original pixspy coordinates
            orig_w, orig_h = self._original_size
            scale_x = self.TARGET_WIDTH / orig_w
            scale_y = self.TARGET_HEIGHT / orig_h

            # Width-based X offset for narrower images (same as name grid)
            REF_WIDTH = 1662
            REF_HEIGHT = 2320
            width_diff = REF_WIDTH - orig_w
            height_diff = REF_HEIGHT - orig_h
            x_offset = int(width_diff * scale_x * 8.0)
            # Height-based Y offset: shorter images need negative offset (move up)
            y_offset = int(height_diff * scale_y * -1.5)

            orig_config = self.ANSWER_BLOCKS_ORIGINAL
            row_gap_orig = orig_config['row_gap']
            option_gap_orig = orig_config['option_gap']
            bubble_size_orig = orig_config['bubble_size']

            # Scale values
            row_gap = int(row_gap_orig * scale_y)
            option_gap = int(option_gap_orig * scale_x)
            bubble_size = (
                int(bubble_size_orig[0] * scale_x),
                int(bubble_size_orig[1] * scale_y)
            )

            # Build scaled answer blocks
            answer_blocks = []
            for block_orig in orig_config['blocks']:
                origin_scaled = (
                    int(block_orig['origin'][0] * scale_x) + x_offset,
                    int(block_orig['origin'][1] * scale_y) + y_offset
                )
                answer_blocks.append({
                    'origin': origin_scaled,
                    'questions': block_orig['questions']
                })

            # Build row positions (Q1 through Q10)
            # First row is Q1, no label row needed with dynamic approach
            base_y = int(orig_config['blocks'][0]['origin'][1] * scale_y)
            row_positions = [base_y + i * row_gap for i in range(10)]

        for block in answer_blocks:
            block_x = block['origin'][0]
            block_y = block['origin'][1] if hasattr(self, '_original_size') else None

            # Per-block offset estimation: search around the calculated position
            # to find the best alignment for this specific block
            if hasattr(self, '_original_size'):
                best_x_off, best_y_off = self._estimate_block_offset_scaled(
                    gray, block_x, block_y, row_gap, option_gap, bubble_size
                )
                block_x += best_x_off
                block_y += best_y_off

            for q_idx, q_num in enumerate(block['questions']):
                # Get Y position for this question
                if hasattr(self, '_original_size'):
                    # Dynamic: block_y + q_idx * row_gap
                    y = block_y + q_idx * row_gap
                else:
                    # Legacy: use row_positions array (with label row at index 0)
                    y = row_positions[q_idx + 1]

                # Check ALL options A, B, C, D to detect multi-marked
                filled_options = []
                for opt_idx, option in enumerate(self.ANSWER_VALUES):
                    # block_x is top-left X of option A
                    x = block_x + opt_idx * option_gap

                    if self._is_bubble_filled_simple(gray, x, y, bubble_size):
                        filled_options.append(option)

                # Determine answer based on filled count
                if len(filled_options) >= 2:
                    # Multiple bubbles filled = MULTI
                    answers[q_num] = 'MULTI'
                elif len(filled_options) == 1:
                    # Single bubble filled = valid answer
                    answers[q_num] = filled_options[0]
                else:
                    # No bubble filled = unanswered
                    answers[q_num] = ''

        return answers

    def _detect_answers_dynamic(self, gray: np.ndarray) -> Dict[int, str]:
        """Detect answers using adaptive grid detection with per-block offset estimation.

        For each answer block, estimates the optimal X/Y offset by searching
        around the calibrated position to find where bubble darkness is maximized.
        This handles alignment variations that differ across the sheet.

        Args:
            gray: Grayscale aligned image

        Returns:
            Dict mapping question number (1-60) to answer (A/B/C/D or empty string)
        """
        answers = {}

        # Calibrated grid parameters
        ROW_GAP = 93  # Vertical spacing between questions
        COL_GAP = 63  # Horizontal spacing between A/B/C/D options

        # Calibrated block origins (baseline)
        BLOCK_CONFIGS = [
            {'origin': (324, 2221), 'questions': list(range(1, 11))},
            {'origin': (660, 2221), 'questions': list(range(11, 21))},
            {'origin': (1058, 2217), 'questions': list(range(21, 31))},
            {'origin': (1396, 2223), 'questions': list(range(31, 41))},
            {'origin': (1795, 2226), 'questions': list(range(41, 51))},
            {'origin': (2163, 2222), 'questions': list(range(51, 61))},
        ]

        for block in BLOCK_CONFIGS:
            origin_x, origin_y = block['origin']
            questions = block['questions']

            # Estimate offset for THIS block specifically
            x_offset, y_offset = self._estimate_block_offset(
                gray, origin_x, origin_y, ROW_GAP, COL_GAP
            )

            # Apply estimated offset
            adj_origin_x = origin_x + x_offset
            adj_origin_y = origin_y + y_offset

            block_answers = self._detect_block_with_grid(
                gray, adj_origin_x, adj_origin_y, ROW_GAP, COL_GAP, questions
            )
            answers.update(block_answers)

        return answers

    def _estimate_block_offset(
        self,
        gray: np.ndarray,
        origin_x: int,
        origin_y: int,
        row_gap: int,
        col_gap: int
    ) -> Tuple[int, int]:
        """Estimate optimal X/Y offset for a single answer block.

        Searches around the calibrated origin position to find the offset
        that maximizes total bubble darkness in the first row. This helps
        handle per-block alignment variations.

        Args:
            gray: Grayscale image
            origin_x: Calibrated X origin for this block
            origin_y: Calibrated Y origin for this block
            row_gap: Row spacing
            col_gap: Column spacing

        Returns:
            (x_offset, y_offset) to adjust this block's origin
        """
        best_x_offset = 0
        best_y_offset = 0
        best_total_darkness = 0

        # Search in a grid around the calibrated position
        for y_off in range(-20, 21, 5):
            for x_off in range(-20, 21, 5):
                total_darkness = 0

                # Sample all 4 options in first row
                for col in range(4):
                    x = origin_x + x_off + col * col_gap
                    y = origin_y + y_off
                    darkness = self._sample_bubble_darkness(gray, x, y, 14)
                    total_darkness += darkness

                if total_darkness > best_total_darkness:
                    best_total_darkness = total_darkness
                    best_x_offset = x_off
                    best_y_offset = y_off

        return (best_x_offset, best_y_offset)

    def _estimate_block_offset_scaled(
        self,
        gray: np.ndarray,
        origin_x: int,
        origin_y: int,
        row_gap: int,
        col_gap: int,
        bubble_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Estimate optimal X/Y offset for a block using scaled coordinates.

        Searches around the calculated origin position to find the offset
        that maximizes the number of clearly filled bubbles (high darkness).
        Uses moderate search range to avoid jumping to wrong grid positions.

        Args:
            gray: Grayscale image
            origin_x: Calculated X origin for this block
            origin_y: Calculated Y origin for this block
            row_gap: Scaled row spacing
            col_gap: Scaled column spacing
            bubble_size: Scaled bubble size (width, height)

        Returns:
            (x_offset, y_offset) to adjust this block's origin
        """
        best_x_offset = 0
        best_y_offset = 0
        best_score = 0

        bw, bh = bubble_size

        # Moderate search range (-30 to +30) to avoid jumping to wrong grids
        for y_off in range(-30, 31, 6):
            for x_off in range(-30, 31, 6):
                score = 0
                filled_count = 0

                # Sample ALL 10 rows to get better coverage
                for row_idx in range(10):
                    y = origin_y + y_off + row_idx * row_gap
                    if y < 0 or y + bh >= gray.shape[0]:
                        continue

                    # Find the darkest option in this row
                    row_max_darkness = 0
                    for col_idx in range(4):
                        x = origin_x + x_off + col_idx * col_gap
                        if x < 0 or x + bw >= gray.shape[1]:
                            continue

                        bubble = gray[y:y+bh, x:x+bw]
                        dark_pixels = np.sum(bubble < 50)
                        if dark_pixels > row_max_darkness:
                            row_max_darkness = dark_pixels

                    # Score based on clear fills (>400 is definitely filled)
                    if row_max_darkness > 400:
                        score += row_max_darkness
                        filled_count += 1

                # Prefer positions with more clearly filled bubbles
                if filled_count > 0:
                    # Normalize by filled count to avoid bias toward more fills
                    normalized_score = score + filled_count * 100
                    if normalized_score > best_score:
                        best_score = normalized_score
                        best_x_offset = x_off
                        best_y_offset = y_off

        # Fine-tune with smaller step around best position
        if best_score > 0:
            coarse_x, coarse_y = best_x_offset, best_y_offset
            for y_off in range(coarse_y - 4, coarse_y + 5, 2):
                for x_off in range(coarse_x - 4, coarse_x + 5, 2):
                    score = 0
                    filled_count = 0

                    for row_idx in range(10):
                        y = origin_y + y_off + row_idx * row_gap
                        if y < 0 or y + bh >= gray.shape[0]:
                            continue

                        row_max_darkness = 0
                        for col_idx in range(4):
                            x = origin_x + x_off + col_idx * col_gap
                            if x < 0 or x + bw >= gray.shape[1]:
                                continue

                            bubble = gray[y:y+bh, x:x+bw]
                            dark_pixels = np.sum(bubble < 50)
                            if dark_pixels > row_max_darkness:
                                row_max_darkness = dark_pixels

                        if row_max_darkness > 400:
                            score += row_max_darkness
                            filled_count += 1

                    if filled_count > 0:
                        normalized_score = score + filled_count * 100
                        if normalized_score > best_score:
                            best_score = normalized_score
                            best_x_offset = x_off
                            best_y_offset = y_off

        return (best_x_offset, best_y_offset)

    def _detect_block_with_grid(
        self,
        gray: np.ndarray,
        origin_x: int,
        origin_y: int,
        row_gap: int,
        col_gap: int,
        questions: List[int]
    ) -> Dict[int, str]:
        """Detect answers using grid sampling at calculated positions.

        Args:
            gray: Grayscale image
            origin_x: X coordinate of first bubble (option A, question 1)
            origin_y: Y coordinate of first bubble
            row_gap: Vertical spacing between questions
            col_gap: Horizontal spacing between options
            questions: List of question numbers

        Returns:
            Dict mapping question number to answer (or 'MULTI' if multiple filled)
        """
        answers = {}

        # Threshold for considering a bubble as filled
        FILLED_THRESHOLD = 50  # Darkness value above which bubble is considered filled

        for row_idx, q_num in enumerate(questions):
            y = int(origin_y + row_idx * row_gap)

            # Sample all 4 options
            darkness_values = []
            for col_idx in range(4):
                x = int(origin_x + col_idx * col_gap)
                darkness = self._sample_bubble_darkness(gray, x, y, 14)
                darkness_values.append(darkness)

            # Find the darkest bubble
            max_darkness = max(darkness_values)
            max_idx = darkness_values.index(max_darkness)

            # Calculate gap between darkest and second darkest
            sorted_darkness = sorted(darkness_values, reverse=True)
            gap = sorted_darkness[0] - sorted_darkness[1] if len(sorted_darkness) > 1 else max_darkness

            # Count how many bubbles are filled (above threshold)
            filled_indices = [i for i, d in enumerate(darkness_values) if d >= FILLED_THRESHOLD]

            # Check for multi-marked: 2+ bubbles filled with similar darkness
            # A bubble is considered filled if darkness >= 50
            # If multiple bubbles are filled and gap between top two is small, it's multi-marked
            if len(filled_indices) >= 2 and gap < 20:
                # Multiple bubbles filled with similar darkness = MULTI
                answers[q_num] = 'MULTI'
            elif max_darkness >= 70:
                # Single clearly filled bubble
                answers[q_num] = 'ABCD'[max_idx]
            elif max_darkness >= 50 and gap >= 5:
                # Filled bubble with clear gap from others
                answers[q_num] = 'ABCD'[max_idx]
            else:
                # No bubble clearly filled
                answers[q_num] = ''

        return answers

    def _detect_block_answers_dynamic(
        self,
        gray: np.ndarray,
        region: Tuple[int, int, int, int],
        questions: List[int]
    ) -> Dict[int, str]:
        """Legacy method - now redirects to grid-based detection."""
        # This method is kept for compatibility but the main detection
        # now uses _detect_answers_dynamic with grid-based approach
        return self._detect_block_answers_calibrated(gray, region, questions)

    def _detect_block_answers_calibrated(
        self,
        gray: np.ndarray,
        region: Tuple[int, int, int, int],
        questions: List[int]
    ) -> Dict[int, str]:
        """Fallback: detect answers using calibrated grid positions."""
        answers = {}

        # Threshold for considering a bubble as filled
        FILLED_THRESHOLD = 50

        # Use the calibrated grid configs from GridDetector
        block_idx = (questions[0] - 1) // 10
        block_names = [
            'answers_1_10', 'answers_11_20', 'answers_21_30',
            'answers_31_40', 'answers_41_50', 'answers_51_60'
        ]

        if block_idx >= len(block_names):
            return {q: '' for q in questions}

        # Get grid config
        if hasattr(self, 'grid_detector') and self.grid_detector:
            grid_config = self.grid_detector.detect_grid(gray, block_names[block_idx])
            if grid_config:
                origin_x, origin_y = grid_config.origin
                row_gap = grid_config.row_gap
                col_gap = grid_config.col_gap

                for row_idx, q_num in enumerate(questions):
                    y = int(origin_y + row_idx * row_gap)

                    darkness_values = []
                    for col_idx in range(4):
                        bx = int(origin_x + col_idx * col_gap)
                        darkness = self._sample_bubble_darkness(gray, bx, y, 14)
                        darkness_values.append(darkness)

                    max_darkness = max(darkness_values)
                    max_idx = darkness_values.index(max_darkness)
                    sorted_darkness = sorted(darkness_values, reverse=True)
                    gap = sorted_darkness[0] - sorted_darkness[1] if len(sorted_darkness) > 1 else max_darkness

                    # Count how many bubbles are filled (above threshold)
                    filled_indices = [i for i, d in enumerate(darkness_values) if d >= FILLED_THRESHOLD]

                    # Check for multi-marked
                    if len(filled_indices) >= 2 and gap < 20:
                        # Multiple bubbles filled with similar darkness = MULTI
                        answers[q_num] = 'MULTI'
                    elif max_darkness >= 70:
                        answers[q_num] = 'ABCD'[max_idx]
                    elif max_darkness >= 50 and gap >= 5:
                        answers[q_num] = 'ABCD'[max_idx]
                    else:
                        answers[q_num] = ''

                return answers

        # Ultimate fallback: return empty
        return {q: '' for q in questions}

    def _cluster_positions(self, values: List[int], tolerance: int = 25) -> List[int]:
        """Cluster 1D positions into groups and return cluster centers."""
        if not values:
            return []

        sorted_vals = sorted(values)
        clusters = [[sorted_vals[0]]]

        for val in sorted_vals[1:]:
            if val - clusters[-1][-1] <= tolerance:
                clusters[-1].append(val)
            else:
                clusters.append([val])

        # Return median of each cluster
        return [int(np.median(c)) for c in clusters]

    def _sample_bubble_darkness(self, gray: np.ndarray, x: int, y: int, radius: int = 14) -> float:
        """Sample darkness at a bubble position."""
        x, y = int(x), int(y)
        r = radius
        x1 = max(0, x - r)
        y1 = max(0, y - r)
        x2 = min(gray.shape[1], x + r)
        y2 = min(gray.shape[0], y + r)

        if x2 <= x1 or y2 <= y1:
            return 0

        region = gray[y1:y2, x1:x2]
        return 255 - np.mean(region)

    def _sample_field(self, gray: np.ndarray, field_name: str) -> str:
        """Sample a field by reading bubble darkness at grid coordinates."""
        # Use hybrid approach if ML classifier is available
        if self.bubble_classifier is not None:
            return self._sample_field_hybrid(gray, field_name)

        return self._sample_field_threshold(gray, field_name)

    def _sample_field_with_confidence(self, gray: np.ndarray, field_name: str) -> Tuple[str, float]:
        """Sample a field and return both value and confidence score.

        Confidence is based on the gap between darkest and second-darkest bubbles:
        - High gap (>40): High confidence (0.9-1.0)
        - Medium gap (20-40): Medium confidence (0.7-0.9)
        - Low gap (<20): Low confidence (0.5-0.7)
        - Very low gap with dark values: Very low confidence (<0.5)
        """
        config = self.field_configs.get(field_name)
        if not config:
            return "", 0.0

        origin = config['origin']
        bubble_size = config['bubble_size']
        rows_gap = config['rows_gap']
        cols_gap = config['cols_gap']
        num_rows = config['rows']
        num_cols = config['cols']
        values = config['values']

        # Debug output for name field
        if field_name == 'name':
            import sys
            print(f"[DEBUG] _sample_field_with_confidence using origin: {origin} for name field", file=sys.stderr)

        preserve_spaces = (field_name == 'name')
        result_chars = []
        column_confidences = []
        consecutive_empty = 0  # Track consecutive empty columns

        # For single-column fields (class, section), search x-offsets to handle alignment variance
        # Different scans can have the grid shifted by 20-30 pixels
        needs_x_search = field_name in ('class', 'section', 'roll_number', 'school_code')
        x_offsets = [-30, -25, -20, -15, -10, -5, 0, 5, 10] if needs_x_search else [0]

        # For name field: calibrate both Y and X offsets for alignment
        # Different scans can have shifts of 10-40 pixels in both directions
        y_offset = 0
        x_offset_calibrated = 0
        if field_name == 'name':
            y_offset = self._calibrate_name_y_offset(gray, origin, bubble_size, rows_gap, cols_gap, num_rows)
            x_offset_calibrated = self._calibrate_name_x_offset(gray, origin, bubble_size, rows_gap, cols_gap, num_rows, y_offset)

        # Get skew correction parameter (name grid is skewed)
        x_shift_per_row = config.get('x_shift_per_row', 0)

        for col in range(num_cols):
            # For name field: check all 25 columns (don't stop early)
            # This allows names like "A<spaces>Z" to be detected correctly
            x_base = origin[0] + col * cols_gap + x_offset_calibrated

            # Search for best x-offset (one with largest gap)
            best_bubble_values = []
            best_gap = -1
            best_darkest_row = -1
            best_darkest_val = 255

            for x_offset in x_offsets:
                # Get all bubble values in this column at this offset
                bubble_values = []
                for row in range(num_rows):
                    # Apply skew correction: X shifts as we go down rows
                    x = int(x_base + x_offset + row * x_shift_per_row)
                    y = origin[1] + y_offset + row * rows_gap  # Apply y_offset for alignment
                    bw, bh = bubble_size
                    if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                        bubble = gray[y:y+bh, x:x+bw]
                        darkness = np.mean(bubble)
                        bubble_values.append((row, darkness))

                if len(bubble_values) >= 2:
                    sorted_bv = sorted(bubble_values, key=lambda x: x[1])
                    gap = sorted_bv[1][1] - sorted_bv[0][1]
                    if gap > best_gap:
                        best_gap = gap
                        best_bubble_values = bubble_values
                        best_darkest_row = sorted_bv[0][0]
                        best_darkest_val = sorted_bv[0][1]

            # Use the best offset results
            bubble_values = best_bubble_values

            if not bubble_values:
                if preserve_spaces:
                    result_chars.append(' ')
                column_confidences.append(0.0)
                continue

            # Analyze bubble values
            all_vals = [v for _, v in bubble_values]
            median_val = np.median(all_vals)
            sorted_bubbles = sorted(bubble_values, key=lambda x: x[1])
            darkest_row, darkest_val = sorted_bubbles[0]
            second_val = sorted_bubbles[1][1] if len(sorted_bubbles) > 1 else median_val
            gap = second_val - darkest_val
            gap_to_median = median_val - darkest_val

            # For name field: use corridor-based detection to distinguish
            # between dark printed letters and actual pencil fills
            corridor_validated = False
            if field_name == 'name':
                # For each candidate bubble, check if the corridor (area between
                # letter and bubble edge) is dark - this proves actual fill vs just dark printing
                candidates_with_corridor = []
                for row, val in bubble_values:
                    if val < 210:  # Only check bubbles that might be filled
                        bubble_y = origin[1] + y_offset + row * rows_gap
                        bubble_x = x_base
                        corridor_filled, corridor_mean = self._check_corridor_fill(
                            gray, bubble_x, bubble_y, bubble_size
                        )
                        candidates_with_corridor.append((row, val, corridor_filled, corridor_mean))

                # Find bubbles that pass corridor check
                corridor_passed = [(r, v, cm) for r, v, cf, cm in candidates_with_corridor if cf]

                if corridor_passed:
                    # Use the FIRST bubble (lowest row number) that passes corridor check
                    # This fixes the issue where multiple bubbles are filled (e.g., both R and U)
                    # and we were picking the darkest instead of the first one
                    corridor_passed.sort(key=lambda x: x[0])  # Sort by row number (first = earliest letter)
                    darkest_row, darkest_val, _ = corridor_passed[0]
                    corridor_validated = True
                else:
                    # No bubbles pass corridor check - might be very light fill
                    # Fall back to checking if darkest bubble has very dark mean
                    if darkest_val < 170:  # Very dark = probably filled
                        corridor_validated = True
                    # Otherwise, column might be empty or have only printed letter darkness

            # Calculate confidence based on gap
            if gap > 50:
                col_conf = 1.0
            elif gap > 40:
                col_conf = 0.95
            elif gap > 30:
                col_conf = 0.85
            elif gap > 20:
                col_conf = 0.75
            elif gap > 12:
                col_conf = 0.65
            elif gap > 8:
                col_conf = 0.55
            else:
                col_conf = 0.4  # Low confidence - ambiguous

            # Detect artifacts
            variance = np.var(all_vals)
            is_artifact_column = variance < 100 and darkest_val < 50

            if is_artifact_column:
                if preserve_spaces:
                    result_chars.append(' ')
                column_confidences.append(0.3)  # Very low confidence
                continue

            # Detection logic for filled bubbles
            # Key insight: A FILLED bubble has darkness typically 40-150
            # An EMPTY bubble has darkness typically 200-230
            # We must check BOTH:
            #   1. Absolute darkness threshold (must be dark enough to be filled)
            #   2. Relative gap (must stand out from other bubbles)
            #
            # Detection thresholds:
            # - Empty bubbles: typically 200-230 darkness, gap_to_median < 15
            # - Filled bubbles: typically 50-180 darkness, gap_to_median > 30
            # - Light fills: 175-195 darkness BUT with strong gap (> 40)
            #
            # Key insight: A strong gap_to_median (> 40) indicates a clearly filled
            # bubble even if the absolute darkness is lighter than usual.

            is_filled = False

            # Primary detection: bubble must be reasonably dark AND stand out
            if darkest_val < 180:  # Normal fills
                if gap_to_median > 30:  # Strong gap = high confidence
                    is_filled = True
                elif gap_to_median > 15 and darkest_val < 160:  # Medium gap, darker fill
                    is_filled = True
                elif gap_to_median > 8 and darkest_val < 140:  # Small gap, very dark fill
                    is_filled = True

            # Light fills with STRONG gap - the gap proves it's filled even if light
            # This catches bubbles in the 180-195 range that clearly stand out
            # IMPORTANT: Require BOTH conditions to avoid false positives:
            #   1. gap_to_median > 40 (stands out from column median)
            #   2. gap > 25 (stands out from second-darkest - not just random variation)
            # The gap > 25 check prevents false positives where one bubble happens to be
            # slightly darker due to print quality but doesn't stand out from other bubbles
            if not is_filled and darkest_val < 195 and gap_to_median > 40 and gap > 25:
                is_filled = True
                col_conf = 0.70  # Slightly lower confidence for light fills

            # Very strong gap_to_median (>45) with reasonable darkness proves fill even with
            # slightly lower gap. The huge gap_to_median means this bubble clearly stands out.
            if not is_filled and darkest_val < 190 and gap_to_median > 45 and gap > 20:
                is_filled = True
                col_conf = 0.65

            # Very light fills (195-205) need even stronger gaps AND second-darkest must be clearly unfilled
            # If second-darkest is also dark (<206), the column has print quality variance, not a real fill
            # NOTE: Don't apply to name field as it causes false positives - name field has its own thresholds
            if not is_filled and darkest_val < 205 and gap_to_median > 40 and gap > 25 and second_val > 210 and field_name != 'name':
                is_filled = True
                col_conf = 0.55

            # For single-column fields (section, class), slightly more lenient
            if not is_filled and field_name in ('section', 'class'):
                if darkest_val < 175 and gap_to_median > 10:
                    is_filled = True
                    col_conf = 0.45
                elif darkest_val < 150 and gap > 5:
                    is_filled = True
                    col_conf = 0.40

            # For name field: use corridor validation as primary check
            # The corridor check examines the area between the printed letter and bubble edge
            # A filled bubble has dark corridor; an unfilled bubble (just printed letter) has light corridor
            if not is_filled and field_name == 'name':
                if corridor_validated:
                    # Corridor check passed - bubble is actually filled
                    is_filled = True
                    col_conf = 0.70
                elif darkest_val < 160:
                    # Very dark bubble - likely filled even if corridor check marginal
                    is_filled = True
                    col_conf = 0.65

            if is_filled:
                if darkest_row < len(values):
                    result_chars.append(values[darkest_row])
                    column_confidences.append(col_conf)
                    consecutive_empty = 0  # Reset counter when we find a filled bubble
            elif preserve_spaces:
                result_chars.append(' ')
                column_confidences.append(1.0)  # Empty column is high confidence
                consecutive_empty += 1  # Track consecutive empty columns
            else:
                column_confidences.append(0.5)  # Uncertain
                consecutive_empty += 1

        result = ''.join(result_chars)
        # Don't normalize spaces - preserve exact spacing as detected
        # if preserve_spaces:
        #     result = ' '.join(result.split())

        # Average confidence across columns (exclude empty columns for non-name fields)
        if column_confidences:
            avg_conf = sum(column_confidences) / len(column_confidences)
        else:
            avg_conf = 0.0

        return result, avg_conf

    def _sample_field_hybrid(self, gray: np.ndarray, field_name: str) -> str:
        """Hybrid approach: threshold for clear cases, ML for uncertain ones."""
        config = self.field_configs.get(field_name)
        if not config:
            return ""

        origin = config['origin']
        bubble_size = config['bubble_size']
        rows_gap = config['rows_gap']
        cols_gap = config['cols_gap']
        num_rows = config['rows']
        num_cols = config['cols']
        values = config['values']

        # Debug output for name field
        if field_name == 'name':
            import sys
            print(f"[DEBUG] _sample_field_hybrid using origin: {origin} for name field", file=sys.stderr)

        preserve_spaces = (field_name == 'name')
        result_chars = []
        consecutive_empty = 0

        # For name field: calibrate Y and X offsets for alignment
        y_offset = 0
        x_offset_calibrated = 0
        if field_name == 'name':
            y_offset = self._calibrate_name_y_offset(gray, origin, bubble_size, rows_gap, cols_gap, num_rows)
            x_offset_calibrated = self._calibrate_name_x_offset(gray, origin, bubble_size, rows_gap, cols_gap, num_rows, y_offset)

        # Get skew correction parameter (name grid is skewed)
        x_shift_per_row = config.get('x_shift_per_row', 0)

        for col in range(num_cols):
            # For name field: check all 25 columns (don't stop early)
            # This allows names like "A<spaces>Z" to be detected correctly

            base_x = origin[0] + col * cols_gap + x_offset_calibrated

            # Extract all bubbles in this column
            bubbles_data = []
            for row in range(num_rows):
                # Apply skew correction: X shifts as we go down rows
                x = int(base_x + row * x_shift_per_row)
                y = origin[1] + y_offset + row * rows_gap
                bw, bh = bubble_size

                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble = gray[y:y+bh, x:x+bw].copy()
                    darkness = np.mean(bubble)
                    bubbles_data.append((row, bubble, darkness))

            if not bubbles_data:
                if preserve_spaces:
                    result_chars.append(' ')
                    consecutive_empty += 1
                continue

            # Sort by darkness to find candidates
            sorted_bubbles = sorted(bubbles_data, key=lambda x: x[2])
            darkest_row, darkest_bubble, darkest_val = sorted_bubbles[0]

            all_vals = [d[2] for d in bubbles_data]
            median_val = np.median(all_vals)
            second_val = sorted_bubbles[1][2] if len(sorted_bubbles) > 1 else median_val
            gap = second_val - darkest_val

            # KEY INSIGHT: A properly filled bubble has a LARGE gap (50+) from unfilled bubbles
            # Poor quality scans with artifacts have SMALL gaps (<30) because multiple bubbles are dark
            #
            # Detection criteria:
            # 1. Gap must be significant (>50 for high confidence, >30 for medium)
            # 2. Darkest must be significantly darker than median
            # 3. Filter out artifact-level darkness (<30) only if gap is also small

            # High confidence: large gap indicates clear filled bubble
            if gap > 50 and darkest_val < median_val * 0.75:
                if darkest_row < len(values):
                    result_chars.append(values[darkest_row])
                    consecutive_empty = 0
                continue

            # Medium confidence: moderate gap with relative darkness check
            if gap > 30 and darkest_val < median_val * 0.65:
                if darkest_row < len(values):
                    result_chars.append(values[darkest_row])
                    consecutive_empty = 0
                continue

            # For single-column fields (section, class) and roll_number, be more lenient before giving up
            # These fields should always try to detect if there's ANY distinguishable filled bubble
            if field_name in ('section', 'class', 'roll_number', 'school_code'):
                gap_to_median = median_val - darkest_val
                # Fallback 1: Lower thresholds for gap and darkness
                if gap > 10 and darkest_val < median_val * 0.80:
                    if darkest_row < len(values):
                        result_chars.append(values[darkest_row])
                        consecutive_empty = 0
                    continue
                # Fallback 2: Very dark bubble (well filled)
                if darkest_val < 150 and gap > 3:
                    if darkest_row < len(values):
                        result_chars.append(values[darkest_row])
                        consecutive_empty = 0
                    continue
                # Fallback 3: Any reasonable gap to median
                if gap_to_median > 5 and darkest_val < 210:
                    if darkest_row < len(values):
                        result_chars.append(values[darkest_row])
                        consecutive_empty = 0
                    continue

            # For name field, also be more lenient to catch lightly filled bubbles
            if field_name == 'name':
                gap_to_median = median_val - darkest_val
                # Fallback: Lower gap threshold for light fills
                if gap_to_median > 15 and darkest_val < 228:
                    if darkest_row < len(values):
                        result_chars.append(values[darkest_row])
                        consecutive_empty = 0
                    continue
                # Fallback 2: Even lighter fills with reasonable gap
                if gap_to_median > 10 and darkest_val < 225:
                    if darkest_row < len(values):
                        result_chars.append(values[darkest_row])
                        consecutive_empty = 0
                    continue

            # Clear case: no bubble is significantly darker (empty column)
            if darkest_val > median_val * 0.92 and gap < 10:
                if preserve_spaces:
                    result_chars.append(' ')
                    consecutive_empty += 1
                continue

            # Uncertain case: use ML classifier
            bubble_images = [b[1] for b in bubbles_data]
            # Resize to 28x28 if needed
            bubble_images = [cv2.resize(b, (28, 28)) if b.shape != (28, 28) else b
                           for b in bubble_images]
            predictions = self.bubble_classifier.predict_batch(bubble_images)

            # Score ALL bubbles and pick the ONE with highest "filled" confidence
            # Tie-breaker: if multiple have same score, pick the FIRST one (lowest row)
            bubble_scores = []
            for (row, bubble_img, darkness), (is_filled, conf) in zip(bubbles_data, predictions):
                filled_score = conf if is_filled else (1 - conf)
                bubble_scores.append((row, filled_score, darkness))

            if bubble_scores:
                # Pick bubble with highest filled_score, then FIRST filled (lowest row), then darkest
                # When multiple bubbles are filled, prefer the first one as that's likely the intended answer
                best_row, best_score, _ = max(bubble_scores, key=lambda x: (x[1], -x[0], -x[2]))

                # Only mark as filled if score is high enough
                # Increased from 0.6 to 0.7 to reduce false positives
                if best_score > 0.7:
                    if best_row < len(values):
                        result_chars.append(values[best_row])
                        consecutive_empty = 0
                elif preserve_spaces:
                    result_chars.append(' ')
                    consecutive_empty += 1
            elif preserve_spaces:
                result_chars.append(' ')
                consecutive_empty += 1

        result = ''.join(result_chars)

        # Don't normalize spaces - preserve exact spacing as detected
        # if preserve_spaces:
        #     result = ' '.join(result.split())

        return result

    def _sample_field_ml(self, gray: np.ndarray, field_name: str) -> str:
        """Sample a field using ML bubble classifier.

        Uses dynamic coordinate scaling based on the actual image dimensions.
        Coordinates from FIELD_CONFIGS_ORIGINAL are scaled by:
            scaled_coord = original_coord * (TARGET_SIZE / original_image_size)
        """
        # Get original config for dynamic scaling
        orig_config = self.FIELD_CONFIGS_ORIGINAL.get(field_name)
        if not orig_config:
            # Fall back to pre-scaled config
            config = self.field_configs.get(field_name)
            if not config:
                return ""
            origin = config['origin']
            bubble_size = config['bubble_size']
            rows_gap = config['rows_gap']
            cols_gap = config['cols_gap']
        else:
            # Dynamic scaling based on actual image dimensions
            REF_WIDTH = 1662  # Reference image width (41_625.jpg)
            REF_HEIGHT = 2320  # Reference image height
            if hasattr(self, '_original_size'):
                orig_w, orig_h = self._original_size
                scale_x = self.TARGET_WIDTH / orig_w
                scale_y = self.TARGET_HEIGHT / orig_h
                # Width-based X offset: narrower images need positive offset (move right)
                width_diff = REF_WIDTH - orig_w
                height_diff = REF_HEIGHT - orig_h
                x_offset = int(width_diff * scale_x * 2.0)  # 2x the scaled difference
                # Height-based Y offset: shorter images need negative offset (move up)
                y_offset_height = int(height_diff * scale_y * -1.5)
            else:
                # Default scale for reference image
                scale_x = self.TARGET_WIDTH / 1662
                scale_y = self.TARGET_HEIGHT / 2320
                x_offset = 0
                y_offset_height = 0

            # Scale all coordinates with width/height-based corrections
            origin = (
                int(orig_config['origin'][0] * scale_x) + x_offset,
                int(orig_config['origin'][1] * scale_y) + y_offset_height
            )
            bubble_size = (
                int(orig_config['bubble_size'][0] * scale_x),
                int(orig_config['bubble_size'][1] * scale_y)
            )
            rows_gap = int(orig_config['rows_gap'] * scale_y)
            cols_gap = int(orig_config.get('cols_gap', orig_config['rows_gap']) * scale_x)

        config = orig_config if orig_config else self.field_configs.get(field_name)
        num_rows = config['rows']
        num_cols = config['cols']
        values = config['values']

        preserve_spaces = (field_name == 'name')
        result_chars = []

        # Get skew correction parameters (scale them if using dynamic scaling)
        if orig_config and hasattr(self, '_original_size'):
            x_shift_per_row = config.get('x_shift_per_row', 0) * scale_x
            y_drift_per_col = config.get('y_drift_per_col', 0) * scale_y
            x_drift_per_col = config.get('x_drift_per_col', 0) * scale_x
            y_drift_per_row = config.get('y_drift_per_row', 0) * scale_y
        else:
            x_shift_per_row = config.get('x_shift_per_row', 0)
            y_drift_per_col = config.get('y_drift_per_col', 0)
            x_drift_per_col = config.get('x_drift_per_col', 0)
            y_drift_per_row = config.get('y_drift_per_row', 0)

        # Apply Y offset calibration for name field (different scans have different alignment)
        y_offset = 0
        if field_name == 'name':
            y_offset = self._calibrate_name_y_offset(gray, origin, bubble_size, rows_gap, cols_gap, num_rows)

        for col in range(num_cols):
            base_x = origin[0] + col * cols_gap + int(col * x_drift_per_col)
            col_y_offset = int(col * y_drift_per_col)  # Y drift for this column

            # Extract all bubbles in this column
            bubbles = []
            for row in range(num_rows):
                # Apply skew correction: X shifts as we go down rows
                x = int(base_x + row * x_shift_per_row)
                y = origin[1] + y_offset + col_y_offset + row * rows_gap + int(row * y_drift_per_row)
                bw, bh = bubble_size

                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble = gray[y:y+bh, x:x+bw].copy()
                    darkness = np.mean(bubble)  # Calculate before resizing
                    if bubble.shape != (28, 28):
                        bubble = cv2.resize(bubble, (28, 28))
                    bubbles.append((row, bubble, darkness))

            if not bubbles:
                if preserve_spaces:
                    result_chars.append(' ')
                continue

            # Use ML classifier to predict each bubble
            bubble_images = [b[1] for b in bubbles]
            predictions = self.bubble_classifier.predict_batch(bubble_images)

            # Score ALL bubbles and pick the ONE with highest "filled" confidence
            # Tie-breaker: if multiple have same score, pick the FIRST one (lowest row = A before B before C)
            # This handles cases where student accidentally filled multiple bubbles in same column
            bubble_scores = []
            for (row, _, darkness), (is_filled, conf) in zip(bubbles, predictions):
                filled_score = conf if is_filled else (1 - conf)
                bubble_scores.append((row, filled_score, darkness))

            if bubble_scores:
                # Find bubbles with high filled score (ML confidence > 0.7)
                # Increased from 0.6 to reduce false positives from stray marks
                filled_candidates = [(r, s, d) for r, s, d in bubble_scores if s > 0.7]

                if filled_candidates:
                    # Always pick FIRST filled bubble (lowest row number)
                    # No fallback to darkest - first filled only
                    filled_candidates.sort(key=lambda x: x[0])  # Sort by row ascending
                    best_row = filled_candidates[0][0]
                    if best_row < len(values):
                        result_chars.append(values[best_row])
                    elif preserve_spaces:
                        result_chars.append(' ')
                elif preserve_spaces:
                    result_chars.append(' ')
            elif preserve_spaces:
                result_chars.append(' ')

        result = ''.join(result_chars)

        if preserve_spaces:
            result = ' '.join(result.split())

        return result

    def _sample_field_threshold(self, gray: np.ndarray, field_name: str) -> str:
        """Sample a field using threshold-based detection (fallback)."""
        config = self.field_configs.get(field_name)
        if not config:
            return ""

        origin = config['origin']
        bubble_size = config['bubble_size']
        rows_gap = config['rows_gap']
        cols_gap = config['cols_gap']
        num_rows = config['rows']
        num_cols = config['cols']
        values = config['values']

        # For name field, preserve spaces between words
        preserve_spaces = (field_name == 'name')
        result_chars = []

        # Get skew correction parameter (name grid is skewed)
        x_shift_per_row = config.get('x_shift_per_row', 0)

        # For each column, find the darkest (filled) bubble
        for col in range(num_cols):
            base_x = origin[0] + col * cols_gap

            # Get all bubble values in this column
            bubble_values = []
            for row in range(num_rows):
                # Apply skew correction: X shifts as we go down rows
                x = int(base_x + row * x_shift_per_row)
                y = origin[1] + row * rows_gap

                # Extract bubble region
                bw, bh = bubble_size
                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble = gray[y:y+bh, x:x+bw]
                    darkness = np.mean(bubble)
                    bubble_values.append((row, darkness))

            if not bubble_values:
                if preserve_spaces:
                    result_chars.append(' ')
                continue

            # Calculate statistics
            all_vals = [v for _, v in bubble_values]
            median_val = np.median(all_vals)
            sorted_bubbles = sorted(bubble_values, key=lambda x: x[1])
            darkest_val = sorted_bubbles[0][1]

            # Detect scan artifacts: if ALL bubbles are similarly dark (low variance),
            # this column might be hitting a dark edge/artifact, not actual bubbles
            variance = np.var(all_vals)
            is_artifact_column = variance < 100 and darkest_val < 50

            if is_artifact_column:
                if preserve_spaces:
                    result_chars.append(' ')
                continue

            # Find ALL bubbles that qualify as "filled" based on threshold criteria
            # Then pick the FIRST one (lowest row number) instead of the darkest
            filled_candidates = []
            for row, darkness in bubble_values:
                bubble_gap_to_median = median_val - darkness

                is_bubble_filled = False

                # Primary detection criteria
                if (bubble_gap_to_median > 40 and darkness < 235):   # Clear filled
                    is_bubble_filled = True
                elif (bubble_gap_to_median > 20 and darkness < 230):  # Good gap, light fill
                    is_bubble_filled = True
                elif (bubble_gap_to_median > 12 and darkness < 220):  # Medium gap
                    is_bubble_filled = True
                elif (bubble_gap_to_median > 8 and darkness < 200):   # Small gap, darker
                    is_bubble_filled = True

                # For single-column fields and roll_number/school_code, be more lenient
                if not is_bubble_filled and field_name in ('section', 'class', 'roll_number', 'school_code'):
                    if bubble_gap_to_median > 5 and darkness < 210:
                        is_bubble_filled = True
                    elif darkness < 150:
                        is_bubble_filled = True
                    elif bubble_gap_to_median > 3 and darkness < median_val * 0.85:
                        is_bubble_filled = True
                    # Additional fallback for very light fills (common in roll_number)
                    # If the bubble is clearly darker than median even if absolute value is high
                    elif bubble_gap_to_median > 8 and darkness < 238:
                        is_bubble_filled = True
                    elif bubble_gap_to_median > 12 and darkness < 242:
                        is_bubble_filled = True

                # For name field, more lenient for light fills
                if not is_bubble_filled and field_name == 'name':
                    if bubble_gap_to_median > 15 and darkness < 228:
                        is_bubble_filled = True
                    elif bubble_gap_to_median > 10 and darkness < 225:
                        is_bubble_filled = True

                if is_bubble_filled:
                    filled_candidates.append((row, darkness))

            # Pick the FIRST filled bubble (lowest row number)
            if filled_candidates:
                filled_candidates.sort(key=lambda x: x[0])  # Sort by row number
                best_row = filled_candidates[0][0]
                if best_row < len(values):
                    result_chars.append(values[best_row])
            elif preserve_spaces:
                # No bubble filled in this column = space between words
                result_chars.append(' ')

        result = ''.join(result_chars)

        # Normalize: collapse multiple spaces into one, strip leading/trailing
        if preserve_spaces:
            result = ' '.join(result.split())

        return result

    def _sample_answers(self, gray: np.ndarray) -> Dict[int, str]:
        """Sample answer bubbles from all answer blocks.

        Uses ML classifier when available, falls back to threshold-based detection.

        The answer section layout:
        - 6 blocks (Q1-10, Q11-20, Q21-30, Q31-40, Q41-50, Q51-60)
        - Questions go VERTICALLY using calibrated row positions
        - Options A,B,C,D go HORIZONTALLY (x increases for A→B→C→D)
        """
        # Use ML-enhanced detection if classifier is available
        if self.answer_classifier is not None and TORCH_AVAILABLE:
            return self._sample_answers_ml(gray)

        # Fall back to threshold-based detection
        return self._sample_answers_threshold(gray)

    def _sample_answers_with_confidence(self, gray: np.ndarray) -> Tuple[Dict[int, str], float]:
        """Sample answers and return both values and confidence score."""
        answers = {}
        question_confidences = []

        # Use batch-specific answer configuration
        answer_blocks = getattr(self, '_current_answer_blocks', self.ANSWER_BLOCKS)
        row_positions = getattr(self, '_current_answer_row_positions', self.ANSWER_ROW_POSITIONS)
        bw, bh = self.ANSWER_BUBBLE_SIZE

        for block in answer_blocks:
            origin_x = block['origin'][0]
            questions = list(block['questions'])

            # IMPORTANT: Find ONE consistent Y offset for the entire block
            # This prevents per-question drift to wrong rows (like label rows or adjacent questions)
            best_block_y_offset = 0
            best_block_x_offset = 0
            best_block_total_gap = 0

            for y_offset in range(-30, 125, 10):
                for x_offset in [-20, -10, 0, 10, 20]:
                    total_gap = 0
                    valid_questions = 0

                    # Calculate total gap across all questions at this offset
                    for q_idx in range(len(questions)):
                        y_base = row_positions[q_idx]
                        y = y_base + y_offset
                        if y < 0 or y >= gray.shape[0] - bh:
                            continue

                        option_values = []
                        for opt_idx in range(4):
                            x = origin_x + x_offset + opt_idx * self.ANSWER_OPTION_GAP
                            if 0 <= x < gray.shape[1] - bw:
                                bubble = gray[y:y+bh, x:x+bw]
                                darkness = np.mean(bubble)
                                option_values.append(darkness)

                        if len(option_values) == 4:
                            sorted_vals = sorted(option_values)
                            gap = sorted_vals[1] - sorted_vals[0]
                            total_gap += gap
                            valid_questions += 1

                    if valid_questions == len(questions) and total_gap > best_block_total_gap:
                        best_block_total_gap = total_gap
                        best_block_y_offset = y_offset
                        best_block_x_offset = x_offset

            # Now detect answers using the consistent block offset
            for q_idx, q_num in enumerate(questions):
                y_base = row_positions[q_idx]
                y = y_base + best_block_y_offset

                option_values = []
                for opt_idx in range(4):
                    x = origin_x + best_block_x_offset + opt_idx * self.ANSWER_OPTION_GAP
                    if 0 <= x < gray.shape[1] - bw and 0 <= y < gray.shape[0] - bh:
                        bubble = gray[y:y+bh, x:x+bw]
                        darkness = np.mean(bubble)
                        option_values.append((opt_idx, darkness))

                if not option_values or len(option_values) != 4:
                    answers[q_num] = ''
                    question_confidences.append(0.0)
                    continue

                sorted_opts = sorted(option_values, key=lambda x: x[1])
                darkest_opt, darkest_val = sorted_opts[0]
                second_val = sorted_opts[1][1] if len(sorted_opts) > 1 else 255
                gap = second_val - darkest_val

                # Calculate confidence based on gap
                if gap > 50:
                    q_conf = 1.0
                elif gap > 40:
                    q_conf = 0.95
                elif gap > 25:
                    q_conf = 0.80
                elif gap > 15:
                    q_conf = 0.65
                elif gap > 10:
                    q_conf = 0.50
                else:
                    q_conf = 0.35  # Low confidence

                # Detection logic - thresholds calibrated for various fill qualities
                # Relaxed thresholds to handle alignment variance across scans
                detected = False
                if gap >= 35 and darkest_val < 200:  # Strong contrast
                    detected = True
                elif gap >= 20 and darkest_val < 190:  # Good contrast
                    detected = True
                elif gap >= 12 and darkest_val < 185:  # Moderate contrast
                    detected = True
                elif gap >= 8 and darkest_val < 175:   # Light fills with some gap
                    detected = True
                elif gap >= 5 and darkest_val < 160:   # Dark fills with small gap
                    detected = True

                # Check for multi-marked
                multi_marked = []
                if detected:
                    threshold = darkest_val + gap * 0.5
                    for opt_idx, opt_val in option_values:
                        if opt_val <= threshold:
                            multi_marked.append(self.ANSWER_VALUES[opt_idx])

                if detected:
                    if len(multi_marked) > 1:
                        answers[q_num] = 'MULTI'  # Use standard MULTI value for scorer
                        q_conf = 0.4  # Multi-marked reduces confidence
                    elif darkest_opt < len(self.ANSWER_VALUES):
                        answers[q_num] = self.ANSWER_VALUES[darkest_opt]
                    else:
                        answers[q_num] = ''
                else:
                    answers[q_num] = ''
                    q_conf = 0.8  # Empty answer is reasonably confident

                question_confidences.append(q_conf)

        # Average confidence across all questions
        if question_confidences:
            avg_conf = sum(question_confidences) / len(question_confidences)
        else:
            avg_conf = 0.0

        return answers, avg_conf

    def _sample_answers_ml(self, gray: np.ndarray) -> Dict[int, str]:
        """Sample answers using ML classifier for improved accuracy."""
        answers = {}
        bw, bh = self.ANSWER_BUBBLE_SIZE

        # Use batch-specific answer configuration
        answer_blocks = getattr(self, '_current_answer_blocks', self.ANSWER_BLOCKS)
        row_positions = getattr(self, '_current_answer_row_positions', self.ANSWER_ROW_POSITIONS)

        for block in answer_blocks:
            origin_x = block['origin'][0]
            questions = list(block['questions'])

            # IMPORTANT: Find ONE consistent Y offset for the entire block
            # This prevents per-question drift to wrong rows (like label rows)
            best_block_offset = 0
            best_block_confidence = 0.0

            for y_offset in range(-30, 125, 15):
                for x_offset in [-20, -10, 0, 10, 20]:
                    total_confidence = 0.0
                    valid_questions = 0

                    # Calculate total confidence across all questions at this offset
                    for q_idx in range(len(questions)):
                        y_base = row_positions[q_idx]
                        y = y_base + y_offset
                        if y < 0 or y >= gray.shape[0] - bh:
                            continue

                        bubbles = []
                        valid = True
                        for opt_idx in range(4):
                            x = origin_x + x_offset + opt_idx * self.ANSWER_OPTION_GAP
                            if 0 <= x < gray.shape[1] - bw:
                                bubble = gray[y:y+bh, x:x+bw]
                                bubble = cv2.resize(bubble, (28, 28))
                                bubbles.append(bubble)
                            else:
                                valid = False
                                break

                        if not valid or len(bubbles) != 4:
                            continue

                        batch = np.stack([b.astype(np.float32) / 255.0 for b in bubbles])
                        batch = batch[:, np.newaxis, :, :]
                        batch_tensor = torch.tensor(batch).to(self.answer_classifier_device)

                        with torch.no_grad():
                            outputs = self.answer_classifier(batch_tensor)
                            probs = torch.softmax(outputs, dim=1)
                            filled_probs = 1 - probs[:, 0]

                        max_prob = filled_probs.max().item()
                        total_confidence += max_prob
                        valid_questions += 1

                    if valid_questions == len(questions) and total_confidence > best_block_confidence:
                        best_block_confidence = total_confidence
                        best_block_offset = y_offset

            # Now detect answers using the consistent block offset
            for q_idx, q_num in enumerate(questions):
                y_base = row_positions[q_idx]
                y = y_base + best_block_offset

                bubbles = []
                valid = True
                for opt_idx in range(4):
                    x = origin_x + opt_idx * self.ANSWER_OPTION_GAP
                    if 0 <= x < gray.shape[1] - bw and 0 <= y < gray.shape[0] - bh:
                        bubble = gray[y:y+bh, x:x+bw]
                        bubble = cv2.resize(bubble, (28, 28))
                        bubbles.append(bubble)
                    else:
                        valid = False
                        break

                if not valid or len(bubbles) != 4:
                    answers[q_num] = ''
                    continue

                batch = np.stack([b.astype(np.float32) / 255.0 for b in bubbles])
                batch = batch[:, np.newaxis, :, :]
                batch_tensor = torch.tensor(batch).to(self.answer_classifier_device)

                with torch.no_grad():
                    outputs = self.answer_classifier(batch_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    filled_probs = 1 - probs[:, 0]

                max_filled_prob, max_idx = filled_probs.max(dim=0)

                if max_filled_prob > 0.5:
                    multi_marked = []
                    for opt_idx in range(4):
                        if filled_probs[opt_idx] > 0.5:
                            multi_marked.append(self.ANSWER_VALUES[opt_idx])

                    if len(multi_marked) > 1:
                        answers[q_num] = 'MULTI'  # Use standard MULTI value for scorer
                    else:
                        answers[q_num] = self.ANSWER_VALUES[max_idx.item()]
                else:
                    answers[q_num] = ''

        return answers

    def _sample_answers_threshold(self, gray: np.ndarray) -> Dict[int, str]:
        """Sample answers using threshold-based detection (fallback)."""
        answers = {}

        # Use batch-specific answer configuration
        answer_blocks = getattr(self, '_current_answer_blocks', self.ANSWER_BLOCKS)
        row_positions = getattr(self, '_current_answer_row_positions', self.ANSWER_ROW_POSITIONS)

        for block in answer_blocks:
            origin_x = block['origin'][0]
            questions = list(block['questions'])
            bw, bh = self.ANSWER_BUBBLE_SIZE

            # IMPORTANT: Find ONE consistent Y offset for the entire block
            # This prevents per-question drift to wrong rows (like label rows)
            best_block_offset = 0
            best_block_total_gap = 0

            for y_offset in range(-30, 125, 10):
                for x_offset in [-20, -10, 0, 10, 20]:
                    total_gap = 0
                    valid_questions = 0

                    # Calculate total gap across all questions at this offset
                    for q_idx in range(len(questions)):
                        y_base = row_positions[q_idx]
                        y = y_base + y_offset
                        if y < 0 or y >= gray.shape[0] - bh:
                            continue

                        option_values = []
                        for opt_idx in range(4):
                            x = origin_x + x_offset + opt_idx * self.ANSWER_OPTION_GAP
                            if 0 <= x < gray.shape[1] - bw:
                                bubble = gray[y:y+bh, x:x+bw]
                                darkness = np.mean(bubble)
                                option_values.append(darkness)

                        if len(option_values) == 4:
                            sorted_vals = sorted(option_values)
                            gap = sorted_vals[1] - sorted_vals[0]
                            total_gap += gap
                            valid_questions += 1

                    if valid_questions == len(questions) and total_gap > best_block_total_gap:
                        best_block_total_gap = total_gap
                        best_block_offset = y_offset

            # Now detect answers using the consistent block offset
            for q_idx, q_num in enumerate(questions):
                y_base = row_positions[q_idx]
                y = y_base + best_block_offset

                option_values = []
                for opt_idx in range(4):
                    x = origin_x + opt_idx * self.ANSWER_OPTION_GAP
                    if 0 <= x < gray.shape[1] - bw and 0 <= y < gray.shape[0] - bh:
                        bubble = gray[y:y+bh, x:x+bw]
                        darkness = np.mean(bubble)
                        option_values.append((opt_idx, darkness))

                if not option_values:
                    answers[q_num] = ''
                    continue

                # Sort by darkness to find candidates
                sorted_opts = sorted(option_values, key=lambda x: x[1])
                darkest_opt, darkest_val = sorted_opts[0]
                second_val = sorted_opts[1][1] if len(sorted_opts) > 1 else 255
                gap = second_val - darkest_val

                # Improved detection thresholds based on calibration:
                # The key insight is that filled bubbles create a clear darkness gap
                # from unfilled ones, regardless of absolute darkness values.
                detected = False

                # Large gap + reasonably dark = definitely filled
                if gap > 40 and darkest_val < 180:
                    detected = True
                # Medium gap + darker = likely filled
                elif gap > 25 and darkest_val < 150:
                    detected = True
                # Small gap + quite dark = might be filled
                elif gap > 15 and darkest_val < 120:
                    detected = True
                # Very small gap but extremely dark = still detect
                elif gap > 10 and darkest_val < 80:
                    detected = True

                # Check for multi-marked (multiple filled bubbles)
                multi_marked = []
                if detected:
                    # Check if other options are also significantly filled
                    threshold = darkest_val + gap * 0.5  # Within half the gap
                    for opt_idx, opt_val in option_values:
                        if opt_val <= threshold:
                            multi_marked.append(self.ANSWER_VALUES[opt_idx])

                if detected:
                    if len(multi_marked) > 1:
                        # Multi-marked: use standard MULTI value for scorer
                        answers[q_num] = 'MULTI'
                    elif darkest_opt < len(self.ANSWER_VALUES):
                        answers[q_num] = self.ANSWER_VALUES[darkest_opt]
                    else:
                        answers[q_num] = ''
                else:
                    answers[q_num] = ''

        return answers

    def _process_with_yolo(self, image: np.ndarray) -> OMRResult:
        """Process using YOLO model for bubble detection."""
        detections = self._detect_with_yolo(image)
        result = self._map_detections_to_fields(detections, image.shape)
        result.all_detections = detections
        return result

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

    def _map_detections_to_fields(
        self,
        detections: List[BubbleDetection],
        image_shape: Tuple[int, int, int]
    ) -> OMRResult:
        """Map YOLO-detected bubbles to form fields based on their position."""

        if not detections:
            return OMRResult(
                success=False,
                error_message="No filled bubbles detected"
            )

        result = OMRResult(success=True)

        # Process each field using FIELD_CONFIGS
        for field_name, config in self.field_configs.items():
            origin = config['origin']
            values = config['values']
            num_cols = config['cols']
            num_rows = config['rows']
            cols_gap = config['cols_gap']
            rows_gap = config['rows_gap']
            bubble_size = config['bubble_size']

            # Calculate region bounds from origin + gaps
            region_x1 = origin[0] - bubble_size[0] // 2
            region_y1 = origin[1] - bubble_size[1] // 2
            region_x2 = origin[0] + num_cols * cols_gap + bubble_size[0]
            region_y2 = origin[1] + num_rows * rows_gap + bubble_size[1]

            # Find detections within this region
            field_detections = [
                d for d in detections
                if region_x1 <= d.x <= region_x2 and region_y1 <= d.y <= region_y2
            ]

            if field_detections:
                # Assign grid positions based on relative position
                for det in field_detections:
                    center_x = det.x + det.width / 2
                    center_y = det.y + det.height / 2
                    det.grid_col = int((center_x - origin[0] + cols_gap / 2) / cols_gap)
                    det.grid_row = int((center_y - origin[1] + rows_gap / 2) / rows_gap)
                    det.grid_col = max(0, min(num_cols - 1, det.grid_col))
                    det.grid_row = max(0, min(num_rows - 1, det.grid_row))

                # Build result by column
                result_chars = []
                for col in range(num_cols):
                    col_dets = [d for d in field_detections if d.grid_col == col]
                    if col_dets:
                        best = min(col_dets, key=lambda d: (d.grid_row, -d.confidence))
                        if best.grid_row < len(values):
                            result_chars.append(values[best.grid_row])
                        else:
                            result_chars.append(' ')
                    else:
                        result_chars.append(' ')

                field_value = ''.join(result_chars).strip()

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

    # Reference image for alignment
    reference_path = "OMRChecker/inputs/VidyaBharti2025/omr-filled.png"

    # Initialize with reference for better alignment
    processor = YOLOOMRProcessor(reference_image_path=reference_path)

    # Test on sample images
    test_images = list(Path("filledomrsheet").glob("*.jpg"))[:5]

    print(f"\n{'Filename':<20} {'Name':<25} {'Roll':<8} {'Class':<8} {'Section':<10}")
    print("-" * 75)

    for img_path in test_images:
        result = processor.process(str(img_path))
        name_display = result.name[:22] + "..." if len(result.name) > 25 else result.name
        print(f"{img_path.name:<20} {name_display:<25} {result.roll_number:<8} "
              f"{result.class_:<8} {result.section:<10}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_processor()

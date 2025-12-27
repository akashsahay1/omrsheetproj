"""
Pure YOLOv8-based OMR processor.

Uses YOLOv8 object detection to directly detect filled bubbles
without relying on fixed coordinate-based grid sampling.

Now includes ML-enhanced answer detection using a trained bubble classifier.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

# Try to import torch for ML classifier
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
    # Calibrated from actual scanned images (41_624.jpg)
    # Origin coordinates point to top-left of first bubble in each field
    FIELD_CONFIGS = {
        'name': {
            # Calibrated 2024-12-27 using circle detection on name bubble grid
            # The name bubble grid is on the left side of the form
            # Each column represents one letter position, rows A-Z for letters
            'origin': (189, 671),
            'bubble_size': (33, 33),
            'rows_gap': 52,  # Vertical distance between A,B,C... rows
            'cols_gap': 54,  # Horizontal distance between letter columns
            'rows': 26,  # A-Z
            'cols': 25,  # Max name length
            'values': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            'direction': 'vertical'
        },
        'roll_number': {
            # Calibrated from actual filled bubbles: col0=1600, col1≈1660, col2≈1720
            'origin': (1600, 1278),
            'bubble_size': (28, 28),
            'rows_gap': 54,
            'cols_gap': 60,  # Average gap ~60px
            'rows': 10,  # 0-9
            'cols': 3,
            'values': list('0123456789'),
            'direction': 'vertical'
        },
        'class': {
            # Calibrated from actual filled bubbles: class '3' at x=1820, row 0
            'origin': (1820, 1278),
            'bubble_size': (28, 28),
            'rows_gap': 54,
            'cols_gap': 56,
            'rows': 8,
            'cols': 1,
            'values': ['3', '4', '5', '6', '7', '8', '9', '10'],
            'direction': 'vertical'
        },
        'section': {
            # Calibrated: x=1960 works across different scan alignments
            'origin': (1960, 1278),
            'bubble_size': (28, 28),
            'rows_gap': 54,
            'cols_gap': 56,
            'rows': 8,
            'cols': 1,
            'values': list('ABCDEFGH'),
            'direction': 'vertical'
        },
        'school_code': {
            # Calibrated from actual filled bubbles: school '0' at x=2135, row 0
            'origin': (2135, 1278),
            'bubble_size': (28, 28),
            'rows_gap': 54,
            'cols_gap': 56,
            'rows': 10,
            'cols': 3,
            'values': list('0123456789'),
            'direction': 'vertical'
        }
    }

    # Answer blocks configuration
    # Answer section: 6 blocks, each with 10 questions arranged VERTICALLY
    # Options A,B,C,D go HORIZONTALLY within each question
    # Coordinates calibrated from actual scanned images (41_624.jpg)
    ANSWER_BLOCKS = [
        {'origin': (270, 2170), 'questions': range(1, 11)},    # Block 1: Q1-10
        {'origin': (630, 2170), 'questions': range(11, 21)},   # Block 2: Q11-20
        {'origin': (990, 2170), 'questions': range(21, 31)},   # Block 3: Q21-30
        {'origin': (1350, 2170), 'questions': range(31, 41)},  # Block 4: Q31-40
        {'origin': (1710, 2170), 'questions': range(41, 51)},  # Block 5: Q41-50
        {'origin': (2070, 2170), 'questions': range(51, 61)},  # Block 6: Q51-60
    ]
    ANSWER_BUBBLE_SIZE = (28, 28)
    ANSWER_OPTION_GAP = 70   # Horizontal gap between A,B,C,D options
    # Row positions are NOT uniform - use empirical values from calibration
    ANSWER_ROW_POSITIONS = [
        2170,  # Q1, Q11, Q21, Q31, Q41, Q51
        2260,  # Q2, Q12, Q22, Q32, Q42, Q52
        2360,  # Q3, Q13, Q23, Q33, Q43, Q53
        2450,  # Q4, Q14, Q24, Q34, Q44, Q54
        2540,  # Q5, Q15, Q25, Q35, Q45, Q55
        2630,  # Q6, Q16, Q26, Q36, Q46, Q56
        2730,  # Q7, Q17, Q27, Q37, Q47, Q57
        2820,  # Q8, Q18, Q28, Q38, Q48, Q58
        2910,  # Q9, Q19, Q29, Q39, Q49, Q59
        3000,  # Q10, Q20, Q30, Q40, Q50, Q60
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

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

        if reference_image_path and Path(reference_image_path).exists():
            self.reference_image = cv2.imread(reference_image_path)

        # Load ML bubble classifier if available
        if bubble_classifier_path and Path(bubble_classifier_path).exists():
            self._load_bubble_classifier(bubble_classifier_path)

        # Load ML answer classifier - try default path if not specified
        if answer_classifier_path is None:
            # Try to find the default model path
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
            image = cv2.imread(image_path)
            if image is None:
                return OMRResult(
                    success=False,
                    error_message=f"Failed to load image: {image_path}"
                )

            # Align image to target dimensions
            aligned = self._align_image(image)

            # Extract fields using grid-based sampling
            if self.model:
                # Use YOLO model for bubble detection
                result = self._process_with_yolo(aligned)
            else:
                # Use direct grid sampling (more reliable)
                result = self._process_with_grid_sampling(aligned)

            # Use OCR fallback if name detection is poor
            result = self._apply_ocr_fallback(result, image_path)

            return result

        except Exception as e:
            import traceback
            return OMRResult(
                success=False,
                error_message=f"Processing error: {str(e)}\n{traceback.format_exc()}"
            )

    def _apply_ocr_fallback(self, result: OMRResult, image_path: str) -> OMRResult:
        """Apply OCR fallback if bubble detection produced poor results."""
        if self.ocr_extractor is None:
            return result

        bubble_name = result.name or ""
        has_space = ' ' in bubble_name
        is_short = len(bubble_name.replace(' ', '')) <= 5

        # Check if name looks suspicious (likely detection failure)
        looks_suspicious = self._is_suspicious_name(bubble_name)

        # Use OCR fallback if:
        # 1. Name is very short (≤5 chars)
        # 2. Name has no space but has characters
        # 3. Name looks suspicious (unusual patterns)
        if is_short or (not has_space and len(bubble_name) > 0) or looks_suspicious:
            try:
                ocr_name, ocr_conf = self.ocr_extractor.extract_name(image_path)

                if ocr_name:
                    ocr_clean = self._clean_ocr_name(ocr_name)

                    # Use OCR if it gives a better result
                    ocr_looks_better = (
                        len(ocr_clean) > len(bubble_name) and
                        ' ' in ocr_clean and
                        not self._is_suspicious_name(ocr_clean)
                    )

                    if ocr_looks_better or (looks_suspicious and ' ' in ocr_clean):
                        result.name = ocr_clean
                        result.ocr_used = True

            except Exception as e:
                pass  # OCR failed, keep bubble result

        return result

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

    def _align_image(self, image: np.ndarray) -> np.ndarray:
        """Align image to target dimensions using simple resize.

        Note: Testing showed that simple resize produces more accurate bubble
        detection than feature-based alignment. Feature alignment was causing
        letter misreads (e.g., 'SHIWANSH_KVMAR_' instead of 'SHIVANSH_KUMAR_').
        """
        h, w = image.shape[:2]

        # If already at target dimensions, return as-is
        if w == self.TARGET_WIDTH and h == self.TARGET_HEIGHT:
            resized = image
        else:
            # Use simple resize - testing showed this is more accurate than feature alignment
            resized = cv2.resize(image, (self.TARGET_WIDTH, self.TARGET_HEIGHT))

        # Skip normalization for now - it may interfere with bubble detection
        # return self._normalize_image(resized)
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

    def _process_with_grid_sampling(self, image: np.ndarray) -> OMRResult:
        """Process OMR sheet using direct grid coordinate sampling."""
        result = OMRResult(success=True)

        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Extract name from bubble grid (A-Z rows, each column = one letter position)
        # The form has BOTH handwritten name (for human readability) AND bubble encoding
        # We use the bubble grid for machine detection
        raw_name, name_conf = self._sample_field_with_confidence(gray, 'name')
        result.name = self._clean_detected_name(raw_name)
        result.field_confidences['name'] = name_conf
        result.ocr_used = False

        result.roll_number, roll_conf = self._sample_field_with_confidence(gray, 'roll_number')
        result.field_confidences['roll_number'] = roll_conf

        result.class_, class_conf = self._sample_field_with_confidence(gray, 'class')
        result.field_confidences['class'] = class_conf

        result.section, section_conf = self._sample_field_with_confidence(gray, 'section')
        result.field_confidences['section'] = section_conf

        result.school_code, school_conf = self._sample_field_with_confidence(gray, 'school_code')
        result.field_confidences['school_code'] = school_conf

        # Extract answers with confidence
        result.answers, answer_conf = self._sample_answers_with_confidence(gray)
        result.field_confidences['answers'] = answer_conf

        # Calculate overall confidence as weighted average
        weights = {
            'name': 0.15,
            'roll_number': 0.15,
            'class': 0.10,
            'section': 0.10,
            'school_code': 0.10,
            'answers': 0.40  # Answers are most important
        }

        total_weight = sum(weights.values())
        weighted_conf = sum(
            result.field_confidences.get(field, 0) * weight
            for field, weight in weights.items()
        )
        result.confidence = round((weighted_conf / total_weight) * 100)  # As percentage

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
            # Remove repeated characters (WWWW -> W, MMM -> M)
            w = re.sub(r'([A-Z])\1{2,}', r'\1', w)
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
        if ' ' in name:
            words = name.split()
            cleaned_words = []
            for word in words:
                cleaned = clean_word(word)
                if cleaned and len(cleaned) >= 2:
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
        if first_name and len(first_name) >= 2:
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
        config = self.FIELD_CONFIGS.get(field_name)
        if not config:
            return "", 0.0

        origin = config['origin']
        bubble_size = config['bubble_size']
        rows_gap = config['rows_gap']
        cols_gap = config['cols_gap']
        num_rows = config['rows']
        num_cols = config['cols']
        values = config['values']

        preserve_spaces = (field_name == 'name')
        result_chars = []
        column_confidences = []
        consecutive_empty = 0  # Track consecutive empty columns

        for col in range(num_cols):
            # For name field: stop after 3+ consecutive empty columns (name has ended)
            if field_name == 'name' and consecutive_empty >= 3:
                break
            x = origin[0] + col * cols_gap

            # Get all bubble values in this column
            bubble_values = []
            for row in range(num_rows):
                y = origin[1] + row * rows_gap
                bw, bh = bubble_size
                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble = gray[y:y+bh, x:x+bw]
                    darkness = np.mean(bubble)
                    bubble_values.append((row, darkness))

            if not bubble_values:
                if preserve_spaces:
                    result_chars.append(' ')
                column_confidences.append(0.0)
                continue

            # Sort bubbles by darkness
            all_vals = [v for _, v in bubble_values]
            median_val = np.median(all_vals)
            sorted_bubbles = sorted(bubble_values, key=lambda x: x[1])
            darkest_row, darkest_val = sorted_bubbles[0]
            second_val = sorted_bubbles[1][1] if len(sorted_bubbles) > 1 else median_val
            gap = second_val - darkest_val
            gap_to_median = median_val - darkest_val

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
            # Use same threshold for all fields - the original calibrated values work well
            is_filled = (
                (gap_to_median > 40 and darkest_val < 235)
                or (gap_to_median > 20 and darkest_val < 230)
                or (gap_to_median > 12 and darkest_val < 220)
                or (gap_to_median > 8 and darkest_val < 200)
            )

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
        if preserve_spaces:
            result = ' '.join(result.split())

        # Average confidence across columns (exclude empty columns for non-name fields)
        if column_confidences:
            avg_conf = sum(column_confidences) / len(column_confidences)
        else:
            avg_conf = 0.0

        return result, avg_conf

    def _sample_field_hybrid(self, gray: np.ndarray, field_name: str) -> str:
        """Hybrid approach: threshold for clear cases, ML for uncertain ones."""
        config = self.FIELD_CONFIGS.get(field_name)
        if not config:
            return ""

        origin = config['origin']
        bubble_size = config['bubble_size']
        rows_gap = config['rows_gap']
        cols_gap = config['cols_gap']
        num_rows = config['rows']
        num_cols = config['cols']
        values = config['values']

        preserve_spaces = (field_name == 'name')
        result_chars = []

        for col in range(num_cols):
            x = origin[0] + col * cols_gap

            # Extract all bubbles in this column
            bubbles_data = []
            for row in range(num_rows):
                y = origin[1] + row * rows_gap
                bw, bh = bubble_size

                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble = gray[y:y+bh, x:x+bw].copy()
                    darkness = np.mean(bubble)
                    bubbles_data.append((row, bubble, darkness))

            if not bubbles_data:
                if preserve_spaces:
                    result_chars.append(' ')
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
                continue

            # Medium confidence: moderate gap with relative darkness check
            if gap > 30 and darkest_val < median_val * 0.65:
                if darkest_row < len(values):
                    result_chars.append(values[darkest_row])
                continue

            # Clear case: no bubble is significantly darker (empty column)
            if darkest_val > median_val * 0.92 and gap < 10:
                if preserve_spaces:
                    result_chars.append(' ')
                continue

            # Uncertain case: use ML classifier
            bubble_images = [b[1] for b in bubbles_data]
            # Resize to 28x28 if needed
            bubble_images = [cv2.resize(b, (28, 28)) if b.shape != (28, 28) else b
                           for b in bubble_images]
            predictions = self.bubble_classifier.predict_batch(bubble_images)

            filled_bubbles = []
            for (row, _, _), (is_filled, conf) in zip(bubbles_data, predictions):
                if is_filled and conf > 0.6:  # Higher confidence threshold
                    filled_bubbles.append((row, conf))

            if filled_bubbles:
                best_row, _ = max(filled_bubbles, key=lambda x: x[1])
                if best_row < len(values):
                    result_chars.append(values[best_row])
            elif preserve_spaces:
                result_chars.append(' ')

        result = ''.join(result_chars)

        if preserve_spaces:
            result = ' '.join(result.split())

        return result

    def _sample_field_ml(self, gray: np.ndarray, field_name: str) -> str:
        """Sample a field using ML bubble classifier."""
        config = self.FIELD_CONFIGS.get(field_name)
        if not config:
            return ""

        origin = config['origin']
        bubble_size = config['bubble_size']
        rows_gap = config['rows_gap']
        cols_gap = config['cols_gap']
        num_rows = config['rows']
        num_cols = config['cols']
        values = config['values']

        preserve_spaces = (field_name == 'name')
        result_chars = []

        for col in range(num_cols):
            x = origin[0] + col * cols_gap

            # Extract all bubbles in this column
            bubbles = []
            for row in range(num_rows):
                y = origin[1] + row * rows_gap
                bw, bh = bubble_size

                if 0 <= y < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                    bubble = gray[y:y+bh, x:x+bw].copy()
                    if bubble.shape != (28, 28):
                        bubble = cv2.resize(bubble, (28, 28))
                    bubbles.append((row, bubble))

            if not bubbles:
                if preserve_spaces:
                    result_chars.append(' ')
                continue

            # Use ML classifier to predict each bubble
            bubble_images = [b[1] for b in bubbles]
            predictions = self.bubble_classifier.predict_batch(bubble_images)

            # Find the filled bubble with highest confidence
            filled_bubbles = []
            for (row, _), (is_filled, conf) in zip(bubbles, predictions):
                if is_filled:
                    filled_bubbles.append((row, conf))

            if filled_bubbles:
                # Pick the one with highest confidence
                best_row, _ = max(filled_bubbles, key=lambda x: x[1])
                if best_row < len(values):
                    result_chars.append(values[best_row])
            elif preserve_spaces:
                result_chars.append(' ')

        result = ''.join(result_chars)

        if preserve_spaces:
            result = ' '.join(result.split())

        return result

    def _sample_field_threshold(self, gray: np.ndarray, field_name: str) -> str:
        """Sample a field using threshold-based detection (fallback)."""
        config = self.FIELD_CONFIGS.get(field_name)
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

        # For each column, find the darkest (filled) bubble
        for col in range(num_cols):
            x = origin[0] + col * cols_gap

            # Get all bubble values in this column
            bubble_values = []
            for row in range(num_rows):
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

            # Sort bubbles by darkness (darkest first)
            all_vals = [v for _, v in bubble_values]
            median_val = np.median(all_vals)
            sorted_bubbles = sorted(bubble_values, key=lambda x: x[1])
            darkest_row, darkest_val = sorted_bubbles[0]

            # Use gap to median (more robust than gap to 2nd darkest which can be affected by bleeding)
            gap_to_median = median_val - darkest_val

            # Detect scan artifacts: if ALL bubbles are similarly dark (low variance),
            # this column might be hitting a dark edge/artifact, not actual bubbles
            variance = np.var(all_vals)
            is_artifact_column = variance < 100 and darkest_val < 50

            if is_artifact_column:
                if preserve_spaces:
                    result_chars.append(' ')
                continue

            # Adaptive detection: bubble is filled if darkest is significantly darker than median
            # Uses gap_to_median which is more robust than gap_to_2nd (handles bleeding)
            # Thresholds tuned for various scan qualities (some have very light fills ~220+)
            is_filled = (
                (gap_to_median > 40 and darkest_val < 235)   # Clear filled bubble
                or (gap_to_median > 20 and darkest_val < 230)  # Good gap, very light fill
                or (gap_to_median > 12 and darkest_val < 220)  # Medium gap, light fill
                or (gap_to_median > 8 and darkest_val < 200)   # Small gap, darker fill
            )

            if is_filled:
                if darkest_row < len(values):
                    result_chars.append(values[darkest_row])
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

        for block in self.ANSWER_BLOCKS:
            origin_x = block['origin'][0]
            questions = list(block['questions'])

            for q_idx, q_num in enumerate(questions):
                y_base = self.ANSWER_ROW_POSITIONS[q_idx]

                # Get darkness values for all options
                option_values = []
                for opt_idx in range(4):
                    x = origin_x + opt_idx * self.ANSWER_OPTION_GAP
                    bw, bh = self.ANSWER_BUBBLE_SIZE
                    if 0 <= y_base < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                        bubble = gray[y_base:y_base+bh, x:x+bw]
                        darkness = np.mean(bubble)
                        option_values.append((opt_idx, darkness))

                if not option_values:
                    answers[q_num] = ''
                    question_confidences.append(0.0)
                    continue

                # Sort by darkness
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

                # Detection logic
                detected = False
                if gap > 40 and darkest_val < 180:
                    detected = True
                elif gap > 25 and darkest_val < 150:
                    detected = True
                elif gap > 15 and darkest_val < 120:
                    detected = True
                elif gap > 10 and darkest_val < 80:
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
                        answers[q_num] = ''.join(multi_marked)
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

        for block in self.ANSWER_BLOCKS:
            origin_x = block['origin'][0]
            questions = list(block['questions'])

            for q_idx, q_num in enumerate(questions):
                y_base = self.ANSWER_ROW_POSITIONS[q_idx]

                # Extract all 4 option bubbles
                bubbles = []
                valid = True
                for opt_idx in range(4):
                    x = origin_x + opt_idx * self.ANSWER_OPTION_GAP
                    if 0 <= y_base < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                        bubble = gray[y_base:y_base+bh, x:x+bw]
                        bubble = cv2.resize(bubble, (28, 28))
                        bubbles.append(bubble)
                    else:
                        valid = False
                        break

                if not valid or len(bubbles) != 4:
                    answers[q_num] = ''
                    continue

                # Run ML classification
                batch = np.stack([b.astype(np.float32) / 255.0 for b in bubbles])
                batch = batch[:, np.newaxis, :, :]  # Add channel dim
                batch_tensor = torch.tensor(batch).to(self.answer_classifier_device)

                with torch.no_grad():
                    outputs = self.answer_classifier(batch_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    # Filled probability = 1 - empty probability
                    filled_probs = 1 - probs[:, 0]

                # Find option with highest filled probability
                max_filled_prob, max_idx = filled_probs.max(dim=0)

                if max_filled_prob > 0.5:
                    # Check for multi-marked
                    multi_marked = []
                    for opt_idx in range(4):
                        if filled_probs[opt_idx] > 0.5:
                            multi_marked.append(self.ANSWER_VALUES[opt_idx])

                    if len(multi_marked) > 1:
                        answers[q_num] = ''.join(multi_marked)
                    else:
                        answers[q_num] = self.ANSWER_VALUES[max_idx.item()]
                else:
                    answers[q_num] = ''

        return answers

    def _sample_answers_threshold(self, gray: np.ndarray) -> Dict[int, str]:
        """Sample answers using threshold-based detection (fallback)."""
        answers = {}

        for block in self.ANSWER_BLOCKS:
            origin_x = block['origin'][0]
            questions = list(block['questions'])

            for q_idx, q_num in enumerate(questions):
                # Use calibrated row positions (non-uniform spacing)
                y_base = self.ANSWER_ROW_POSITIONS[q_idx]

                # Get darkness values for all options (A, B, C, D)
                option_values = []
                for opt_idx in range(4):
                    # Options go horizontally - x changes for A,B,C,D
                    x = origin_x + opt_idx * self.ANSWER_OPTION_GAP

                    bw, bh = self.ANSWER_BUBBLE_SIZE
                    if 0 <= y_base < gray.shape[0] - bh and 0 <= x < gray.shape[1] - bw:
                        bubble = gray[y_base:y_base+bh, x:x+bw]
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
                        # Multi-marked: record all marked options
                        answers[q_num] = ''.join(multi_marked)
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
        for field_name, config in self.FIELD_CONFIGS.items():
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

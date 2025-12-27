"""
OCR-based name extraction from handwritten region.

Uses EasyOCR (preferred) or Tesseract OCR to read the handwritten name
from OMR sheets.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import re

# Try to import EasyOCR (preferred for handwriting)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Fallback to Tesseract
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class OCRNameExtractor:
    """Extract handwritten names using OCR."""

    # Target dimensions (same as YOLOOMRProcessor)
    TARGET_WIDTH = 2480
    TARGET_HEIGHT = 3507

    # Handwritten name region coordinates (after resize to target dimensions)
    # This is the area where the student writes their name in block letters
    # Found via calibration: name is in individual letter boxes at y=590-660
    # (below the "CANDIDATE'S NAME" label which is around y=550)
    NAME_REGION = {
        'x': 30,
        'y': 595,
        'width': 950,  # Extended to capture full names (up to ~25 letters)
        'height': 65  # Covers the handwritten letter row only
    }

    def __init__(self, backend: str = 'auto'):
        """
        Initialize the OCR extractor.

        Args:
            backend: 'easyocr', 'tesseract', or 'auto' (default)
        """
        self.backend = backend
        self._easyocr_reader = None

        # Auto-select backend
        if backend == 'auto':
            if EASYOCR_AVAILABLE:
                self.backend = 'easyocr'
            elif TESSERACT_AVAILABLE:
                self.backend = 'tesseract'
            else:
                raise RuntimeError("No OCR backend available. Install easyocr or pytesseract.")

        # Configure Tesseract
        self.tesseract_config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def _get_easyocr_reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._easyocr_reader is None:
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
        return self._easyocr_reader

    def extract_name(self, image_path: str) -> Tuple[str, float]:
        """
        Extract handwritten name from OMR sheet.

        Args:
            image_path: Path to the OMR image

        Returns:
            Tuple of (extracted_name, confidence)
        """
        # Load and resize image
        image = cv2.imread(image_path)
        if image is None:
            return "", 0.0

        aligned = cv2.resize(image, (self.TARGET_WIDTH, self.TARGET_HEIGHT))

        # Extract name region
        region = self.NAME_REGION
        name_roi = aligned[region['y']:region['y']+region['height'],
                          region['x']:region['x']+region['width']]

        # Preprocess for OCR
        processed = self._preprocess_for_ocr(name_roi)

        # Run OCR based on backend
        if self.backend == 'easyocr':
            return self._extract_with_easyocr(name_roi, processed)
        else:
            return self._extract_with_tesseract(processed)

    def _extract_with_easyocr(self, original_roi: np.ndarray, processed: np.ndarray) -> Tuple[str, float]:
        """Extract text using EasyOCR."""
        try:
            reader = self._get_easyocr_reader()

            # Since letters are in individual boxes, we need to read each letter
            # and join them together. Use detail mode to get character-level info.
            all_texts = []

            # Try with original ROI (color) - EasyOCR handles colors well
            result1 = reader.readtext(original_roi, detail=1, paragraph=False)
            if result1:
                # Sort by x-coordinate to get left-to-right order
                result1.sort(key=lambda x: x[0][0][0])  # Sort by leftmost x
                for item in result1:
                    if len(item) >= 2:
                        text = item[1] if isinstance(item[1], str) else str(item[1])
                        all_texts.append(text)

            # Join all detected text pieces
            combined = ''.join(all_texts)

            # Clean the text - only keep letters
            clean_text = re.sub(r'[^A-Za-z]', '', combined).upper()

            # Add spaces between name parts (detect word boundaries)
            # Common Indian name patterns: FIRSTNAME + KUMAR/KUMARI + SURNAME
            # Insert space before common name suffixes
            for suffix in ['KUMAR', 'KUMARI', 'SINGH', 'DAS', 'PAL', 'KHAN', 'MANDAL', 'YADAV', 'SHARMA', 'RAWANI', 'RAJWAR', 'MAHTO', 'TURI', 'PRIYA', 'SEN', 'SIMHA', 'MARANDI', 'PRAKHAR', 'KUSHWAHA', 'ARFIN', 'SONI', 'RAJA', 'NAHTO']:
                if suffix in clean_text and not clean_text.startswith(suffix):
                    # Insert space before the suffix
                    idx = clean_text.find(suffix)
                    if idx > 0:
                        clean_text = clean_text[:idx] + ' ' + clean_text[idx:]

            # Normalize spaces
            clean_text = ' '.join(clean_text.split())

            if clean_text and len(clean_text) >= 3:
                # Estimate confidence
                conf = 0.6 if len(clean_text) > 5 else 0.4
                return clean_text, conf
            else:
                return "", 0.0

        except Exception as e:
            print(f"EasyOCR error: {e}")
            return "", 0.0

    def _extract_with_tesseract(self, processed: np.ndarray) -> Tuple[str, float]:
        """Extract text using Tesseract."""
        try:
            raw_text = pytesseract.image_to_string(
                processed,
                config=self.tesseract_config
            )

            # Clean the text - only keep letters and spaces
            clean_text = re.sub(r'[^A-Za-z\s]', '', raw_text).upper()
            clean_text = ' '.join(clean_text.split())  # Normalize spaces

            if clean_text and len(clean_text) >= 3:
                # Estimate confidence based on character count
                conf = min(1.0, len(clean_text) / 15.0)
                return clean_text, conf
            else:
                return "", 0.0

        except Exception as e:
            print(f"Tesseract error: {e}")
            return "", 0.0

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results on handwritten text in boxes."""
        if len(image.shape) != 3:
            # Already grayscale
            gray = image
        else:
            # The grid lines are pink/red. Remove them by:
            # 1. Convert to HSV to detect pink/red
            # 2. Create mask for non-pink areas
            # 3. Keep only the blue/black handwritten text

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Pink/red hue range (0-10 and 160-180 in OpenCV)
            # Create mask for pink grid lines
            lower_pink1 = np.array([0, 30, 100])
            upper_pink1 = np.array([15, 255, 255])
            lower_pink2 = np.array([160, 30, 100])
            upper_pink2 = np.array([180, 255, 255])

            mask_pink1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
            mask_pink2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
            mask_pink = cv2.bitwise_or(mask_pink1, mask_pink2)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Remove pink areas by setting them to white
            gray[mask_pink > 0] = 255

        # Scale up for better OCR
        scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # Apply Gaussian blur to smooth
        blurred = cv2.GaussianBlur(scaled, (3, 3), 0)

        # Otsu's thresholding - good for bimodal images
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert if text is white on black (we want black text on white)
        if np.mean(binary) < 128:
            binary = cv2.bitwise_not(binary)

        return binary

    def extract_name_with_debug(
        self,
        image_path: str,
        debug_dir: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Extract name with debug output.

        Args:
            image_path: Path to the OMR image
            debug_dir: Directory to save debug images

        Returns:
            Tuple of (extracted_name, confidence)
        """
        image = cv2.imread(image_path)
        if image is None:
            return "", 0.0

        aligned = cv2.resize(image, (self.TARGET_WIDTH, self.TARGET_HEIGHT))

        region = self.NAME_REGION
        name_roi = aligned[region['y']:region['y']+region['height'],
                          region['x']:region['x']+region['width']]

        processed = self._preprocess_for_ocr(name_roi)

        # Save debug images
        if debug_dir:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)

            img_name = Path(image_path).stem
            cv2.imwrite(str(debug_path / f'{img_name}_roi.jpg'), name_roi)
            cv2.imwrite(str(debug_path / f'{img_name}_processed.jpg'), processed)

        # Run OCR based on backend
        if self.backend == 'easyocr':
            return self._extract_with_easyocr(name_roi, processed)
        else:
            return self._extract_with_tesseract(processed)


def test_ocr():
    """Test OCR extraction on sample images."""
    extractor = OCRNameExtractor()
    print(f"Using OCR backend: {extractor.backend}")

    # Use cross-platform paths relative to this file
    project_dir = Path(__file__).parent.parent.parent  # Go up from src/ai to project root
    test_dir = project_dir / 'filledomrsheet'
    debug_dir = str(project_dir / 'ocr_debug')

    # Test on a few images
    test_images = [
        '41_629.jpg',  # REYANSH KUMAR
        '41_632.jpg',  # YUVRAJ KUMAR MANDAL
        '41_635.jpg',  # ANJALI KUMARI
        '41_637.jpg',  # ROSHAN RAJWAR
    ]

    print("OCR Name Extraction Test")
    print("=" * 60)

    for img_name in test_images:
        img_path = test_dir / img_name
        if img_path.exists():
            name, conf = extractor.extract_name_with_debug(
                str(img_path),
                debug_dir
            )
            print(f"{img_name:<20} -> '{name}' (conf: {conf:.2f})")

    print("=" * 60)
    print(f"Debug images saved to: {debug_dir}")


if __name__ == "__main__":
    test_ocr()

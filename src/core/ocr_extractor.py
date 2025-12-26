"""
OCR-based extraction for OMR sheets using EasyOCR.
Extracts handwritten name and validates bubble detection.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger('omr_processor.ocr_extractor')

# Lazy load EasyOCR to avoid slow startup
_ocr_reader = None

def get_ocr_reader():
    """Get or initialize EasyOCR reader (lazy loading)."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            logger.info("Initializing EasyOCR reader (first time, may take a moment)...")
            _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            return None
    return _ocr_reader


class OCRExtractor:
    """
    Extract text from OMR sheets using OCR.

    This class handles:
    1. Handwritten name extraction from block letter area
    2. Validation of bubble-detected values
    """

    # Region coordinates for 1360x1928 template
    # These define where handwritten text appears on the OMR sheet
    REGIONS = {
        'candidate_name': {
            'x': 30, 'y': 100, 'width': 420, 'height': 55,
            'description': 'Handwritten candidate name in block letters'
        },
        'school_name': {
            'x': 770, 'y': 290, 'width': 300, 'height': 35,
            'description': 'School name field'
        },
        'exam_date': {
            'x': 770, 'y': 325, 'width': 300, 'height': 30,
            'description': 'Examination date field'
        }
    }

    def __init__(self, template_dimensions: Tuple[int, int] = (1360, 1928)):
        """
        Initialize OCR extractor.

        Args:
            template_dimensions: Expected (width, height) of OMR template
        """
        self.template_width, self.template_height = template_dimensions
        self._reader = None

    def _get_reader(self):
        """Get OCR reader instance."""
        if self._reader is None:
            self._reader = get_ocr_reader()
        return self._reader

    def _resize_to_template(self, image: np.ndarray) -> np.ndarray:
        """Resize image to match template dimensions."""
        h, w = image.shape[:2]
        if w != self.template_width or h != self.template_height:
            image = cv2.resize(image, (self.template_width, self.template_height))
        return image

    def _crop_region(self, image: np.ndarray, region_name: str) -> Optional[np.ndarray]:
        """Crop a specific region from the image."""
        if region_name not in self.REGIONS:
            logger.warning(f"Unknown region: {region_name}")
            return None

        region = self.REGIONS[region_name]
        x, y, w, h = region['x'], region['y'], region['width'], region['height']

        # Ensure bounds are within image
        img_h, img_w = image.shape[:2]
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        w = min(w, img_w - x)
        h = min(h, img_h - y)

        return image[y:y+h, x:x+w]

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image region for better OCR accuracy."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply adaptive thresholding for better text contrast
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Slight blur to reduce noise
        denoised = cv2.medianBlur(binary, 3)

        # Scale up for better OCR (EasyOCR works better with larger images)
        scale_factor = 2
        scaled = cv2.resize(denoised, None, fx=scale_factor, fy=scale_factor,
                          interpolation=cv2.INTER_CUBIC)

        return scaled

    def extract_handwritten_name(self, image_path: str) -> Optional[str]:
        """
        Extract handwritten name from the OMR sheet.

        Args:
            image_path: Path to OMR image file

        Returns:
            Extracted name string or None if extraction failed
        """
        reader = self._get_reader()
        if reader is None:
            logger.error("OCR reader not available")
            return None

        try:
            # Load and resize image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            image = self._resize_to_template(image)

            # Crop the name region
            name_region = self._crop_region(image, 'candidate_name')
            if name_region is None:
                return None

            # Preprocess for OCR
            processed = self._preprocess_for_ocr(name_region)

            # Run OCR
            results = reader.readtext(processed, detail=0, paragraph=True)

            if results:
                # Join all detected text and clean up
                name = ' '.join(results).strip().upper()
                # Remove common OCR artifacts
                name = self._clean_ocr_text(name)
                logger.info(f"OCR extracted name: '{name}'")
                return name

            logger.warning("No text detected by OCR")
            return None

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return None

    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR-extracted text."""
        import re

        # Remove non-alphabetic characters except spaces
        cleaned = re.sub(r'[^A-Z\s]', '', text)

        # Normalize whitespace
        cleaned = ' '.join(cleaned.split())

        return cleaned

    def extract_all_text_fields(self, image_path: str) -> Dict[str, str]:
        """
        Extract all handwritten text fields from OMR sheet.

        Args:
            image_path: Path to OMR image file

        Returns:
            Dictionary with extracted text for each field
        """
        results = {}

        reader = self._get_reader()
        if reader is None:
            return results

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return results

            image = self._resize_to_template(image)

            for region_name in self.REGIONS:
                region_img = self._crop_region(image, region_name)
                if region_img is not None:
                    processed = self._preprocess_for_ocr(region_img)
                    ocr_results = reader.readtext(processed, detail=0, paragraph=True)
                    if ocr_results:
                        text = ' '.join(ocr_results).strip()
                        if region_name == 'candidate_name':
                            text = self._clean_ocr_text(text.upper())
                        results[region_name] = text

            return results

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return results


def extract_name_with_ocr(image_path: str) -> Optional[str]:
    """
    Convenience function to extract just the handwritten name.

    Args:
        image_path: Path to OMR image file

    Returns:
        Extracted name or None
    """
    extractor = OCRExtractor()
    return extractor.extract_handwritten_name(image_path)


# Test function
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = r"F:\Projects\omrsheetproj\filledomrsheet\41_624.jpg"

    print(f"Testing OCR on: {image_path}")

    extractor = OCRExtractor()

    # Test single name extraction
    name = extractor.extract_handwritten_name(image_path)
    print(f"Extracted name: {name}")

    # Test all fields
    all_fields = extractor.extract_all_text_fields(image_path)
    print(f"All fields: {all_fields}")

"""
Robust Class/Section Detector with Multi-Method Detection

This module provides a robust detector for Class and Section fields
that combines multiple detection methods:
1. ViT-based classification (when trained model available)
2. Adaptive threshold detection with gap analysis
3. Handwritten OCR fallback (when bubble detection fails)

This addresses the issue where 6-7 out of 26 sheets showed blank
Class/Section even though students filled correctly.
"""

import os
import numpy as np
import cv2
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class RobustClassSectionDetector:
    """
    Multi-method detector for Class and Section fields.

    Uses a cascading approach:
    1. Primary: Adaptive threshold with enhanced gap analysis
    2. Secondary: ViT classifier (if available)
    3. Tertiary: Handwritten OCR fallback
    4. Ensemble voting with confidence weighting
    """

    # Field configurations (coordinates for 2480x3507 normalized image)
    FIELD_CONFIGS = {
        'class': {
            'origin': (1820, 1278),
            'rows': 8,
            'cols': 1,
            'row_gap': 54,
            'bubble_size': (28, 28),
            'values': ['3', '4', '5', '6', '7', '8', '9', '10'],
            # Handwritten box location (approximate)
            'handwritten_box': (1760, 1210, 80, 50)  # x, y, w, h
        },
        'section': {
            'origin': (1960, 1278),
            'rows': 8,
            'cols': 1,
            'row_gap': 54,
            'bubble_size': (28, 28),
            'values': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            # Handwritten box location (approximate)
            'handwritten_box': (1900, 1210, 80, 50)  # x, y, w, h
        }
    }

    # Target image size
    TARGET_SIZE = (2480, 3507)

    def __init__(self, vit_model=None, ocr_extractor=None):
        """
        Initialize the detector.

        Args:
            vit_model: Optional ViT classifier instance
            ocr_extractor: Optional OCR extractor for handwritten text
        """
        self.vit_model = vit_model
        self.ocr_extractor = ocr_extractor

        # Detection thresholds
        self.gap_threshold_high = 40  # Definite detection
        self.gap_threshold_medium = 25  # Probable detection
        self.gap_threshold_low = 15  # Possible detection

        # Confidence thresholds for fallback
        self.confidence_threshold = 0.5

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size if needed."""
        if image.shape[:2] != self.TARGET_SIZE[::-1]:
            image = cv2.resize(image, self.TARGET_SIZE)
        return image

    def extract_bubbles(self, gray: np.ndarray, field: str) -> List[Dict]:
        """
        Extract all bubble regions for a field.

        Args:
            gray: Grayscale image (normalized size)
            field: 'class' or 'section'

        Returns:
            List of bubble data with positions and intensities
        """
        config = self.FIELD_CONFIGS[field]
        origin = config['origin']
        rows = config['rows']
        row_gap = config['row_gap']
        bubble_size = config['bubble_size']
        values = config['values']

        bubbles = []
        for row in range(rows):
            x = origin[0]
            y = origin[1] + row * row_gap

            # Extract bubble region with padding
            padding = 4
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(gray.shape[1], x + bubble_size[0] + padding)
            y2 = min(gray.shape[0], y + bubble_size[1] + padding)

            region = gray[y1:y2, x1:x2]

            # Calculate statistics
            mean_intensity = np.mean(region)
            std_intensity = np.std(region)

            # Calculate fill ratio (dark pixels)
            threshold = mean_intensity * 0.7
            dark_pixels = np.sum(region < threshold)
            fill_ratio = dark_pixels / region.size

            bubbles.append({
                'row': row,
                'value': values[row],
                'x': x,
                'y': y,
                'region': region,
                'mean_intensity': mean_intensity,
                'std_intensity': std_intensity,
                'fill_ratio': fill_ratio
            })

        return bubbles

    def detect_with_threshold(self, bubbles: List[Dict]) -> Tuple[Optional[str], float]:
        """
        Detect filled bubble using adaptive threshold method.

        Args:
            bubbles: List of bubble data

        Returns:
            (detected_value, confidence) or (None, 0) if no detection
        """
        if not bubbles:
            return None, 0.0

        # Sort by darkness (lower intensity = darker = more likely filled)
        sorted_bubbles = sorted(bubbles, key=lambda b: b['mean_intensity'])

        darkest = sorted_bubbles[0]
        second_darkest = sorted_bubbles[1] if len(sorted_bubbles) > 1 else None

        # Calculate gap
        gap = 0
        if second_darkest:
            gap = second_darkest['mean_intensity'] - darkest['mean_intensity']

        # Determine confidence based on gap
        if gap >= self.gap_threshold_high:
            confidence = 0.95
        elif gap >= self.gap_threshold_medium:
            confidence = 0.75
        elif gap >= self.gap_threshold_low:
            confidence = 0.55
        else:
            confidence = 0.3

        # Additional check: fill ratio
        if darkest['fill_ratio'] < 0.15:
            # Very low fill ratio - might be unfilled
            confidence *= 0.5
        elif darkest['fill_ratio'] > 0.4:
            # High fill ratio - boost confidence
            confidence = min(confidence * 1.2, 0.99)

        # Additional check: absolute darkness
        if darkest['mean_intensity'] > 180:
            # Very light - probably not filled
            confidence *= 0.3
        elif darkest['mean_intensity'] < 100:
            # Very dark - boost confidence
            confidence = min(confidence * 1.1, 0.99)

        # Return result
        if confidence >= 0.4:
            return darkest['value'], confidence
        else:
            return None, confidence

    def detect_with_vit(self, bubbles: List[Dict]) -> Tuple[Optional[str], float]:
        """
        Detect filled bubble using ViT classifier.

        Args:
            bubbles: List of bubble data

        Returns:
            (detected_value, confidence) or (None, 0) if no detection
        """
        if self.vit_model is None:
            return None, 0.0

        try:
            # Prepare images for batch prediction
            images = [b['region'] for b in bubbles]
            predictions = self.vit_model.predict_batch(images)

            # Find the bubble with highest fill probability
            best_idx = -1
            best_prob = 0.0

            for i, pred in enumerate(predictions):
                if pred['is_filled'] and pred['is_filled_prob'] > best_prob:
                    best_idx = i
                    best_prob = pred['is_filled_prob']

            if best_idx >= 0 and best_prob > 0.5:
                return bubbles[best_idx]['value'], best_prob
            else:
                return None, 0.0

        except Exception as e:
            print(f"ViT detection error: {e}")
            return None, 0.0

    def detect_with_ocr(self, gray: np.ndarray, field: str) -> Tuple[Optional[str], float]:
        """
        Detect value from handwritten text using OCR.

        Args:
            gray: Grayscale image
            field: 'class' or 'section'

        Returns:
            (detected_value, confidence) or (None, 0) if no detection
        """
        if self.ocr_extractor is None:
            return None, 0.0

        config = self.FIELD_CONFIGS[field]
        box = config['handwritten_box']
        valid_values = config['values']

        try:
            # Extract handwritten region
            x, y, w, h = box
            region = gray[y:y+h, x:x+w]

            # Use OCR to extract text
            if hasattr(self.ocr_extractor, 'extract_text'):
                text, confidence = self.ocr_extractor.extract_text(region)
            else:
                # Fallback to EasyOCR if available
                text, confidence = self._simple_ocr(region, field)

            if text:
                # Clean and validate text
                text = text.strip().upper()

                # For class, try to match number
                if field == 'class':
                    # Extract digits
                    digits = ''.join(c for c in text if c.isdigit())
                    if digits and digits in valid_values:
                        return digits, confidence
                    # Try matching common OCR errors
                    mapping = {'O': '0', 'l': '1', 'I': '1', 'S': '5', 'B': '8'}
                    cleaned = ''.join(mapping.get(c, c) for c in text if c.isdigit() or c in mapping)
                    if cleaned and cleaned in valid_values:
                        return cleaned, confidence * 0.8

                # For section, try to match letter
                else:
                    for char in text:
                        if char in valid_values:
                            return char, confidence
                    # Try matching common OCR errors
                    mapping = {'0': 'O', '8': 'B', '1': 'I'}
                    for char in text:
                        mapped = mapping.get(char, char)
                        if mapped in valid_values:
                            return mapped, confidence * 0.8

            return None, 0.0

        except Exception as e:
            print(f"OCR detection error: {e}")
            return None, 0.0

    def _simple_ocr(self, region: np.ndarray, field: str) -> Tuple[str, float]:
        """
        Simple OCR fallback using template matching or EasyOCR.

        Args:
            region: Image region to OCR
            field: 'class' or 'section'

        Returns:
            (text, confidence)
        """
        try:
            import easyocr
            reader = easyocr.Reader(['en'], gpu=False)

            # Preprocess for better OCR
            if len(region.shape) == 2:
                region_rgb = cv2.cvtColor(region, cv2.COLOR_GRAY2RGB)
            else:
                region_rgb = region

            # Run OCR
            results = reader.readtext(region_rgb)

            if results:
                text = results[0][1]
                confidence = results[0][2]
                return text, confidence

        except ImportError:
            pass
        except Exception as e:
            print(f"EasyOCR error: {e}")

        return "", 0.0

    def detect(self, image: np.ndarray, field: str) -> Dict:
        """
        Detect Class or Section value using multi-method approach.

        Args:
            image: Input image (BGR or grayscale)
            field: 'class' or 'section'

        Returns:
            Dictionary with detection result:
            {
                'value': detected value or None,
                'confidence': float (0-1),
                'method': detection method used,
                'all_results': results from all methods
            }
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Normalize size
        gray = self.normalize_image(gray)

        # Extract bubbles
        bubbles = self.extract_bubbles(gray, field)

        # Method 1: Threshold-based detection
        threshold_value, threshold_conf = self.detect_with_threshold(bubbles)

        # Method 2: ViT-based detection (if available)
        vit_value, vit_conf = self.detect_with_vit(bubbles)

        # Method 3: OCR fallback (if primary methods fail)
        ocr_value, ocr_conf = None, 0.0
        if threshold_conf < self.confidence_threshold and vit_conf < self.confidence_threshold:
            ocr_value, ocr_conf = self.detect_with_ocr(gray, field)

        # Collect all results
        all_results = {
            'threshold': {'value': threshold_value, 'confidence': threshold_conf},
            'vit': {'value': vit_value, 'confidence': vit_conf},
            'ocr': {'value': ocr_value, 'confidence': ocr_conf}
        }

        # Ensemble: weighted voting
        candidates = {}

        if threshold_value:
            weight = threshold_conf * 1.0  # Base weight
            candidates[threshold_value] = candidates.get(threshold_value, 0) + weight

        if vit_value:
            weight = vit_conf * 1.2  # ViT slightly more trusted
            candidates[vit_value] = candidates.get(vit_value, 0) + weight

        if ocr_value:
            weight = ocr_conf * 0.8  # OCR slightly less trusted
            candidates[ocr_value] = candidates.get(ocr_value, 0) + weight

        # Select best candidate
        if candidates:
            best_value = max(candidates, key=candidates.get)
            best_score = candidates[best_value]

            # Determine which method contributed most
            if vit_value == best_value and vit_conf >= threshold_conf and vit_conf >= ocr_conf:
                method = 'vit'
                confidence = vit_conf
            elif threshold_value == best_value and threshold_conf >= ocr_conf:
                method = 'threshold'
                confidence = threshold_conf
            else:
                method = 'ocr'
                confidence = ocr_conf

            # Boost confidence if multiple methods agree
            agreeing_methods = sum([
                1 if threshold_value == best_value else 0,
                1 if vit_value == best_value else 0,
                1 if ocr_value == best_value else 0
            ])
            if agreeing_methods >= 2:
                confidence = min(confidence * 1.2, 0.99)
                method = 'ensemble'

            return {
                'value': best_value,
                'confidence': confidence,
                'method': method,
                'all_results': all_results
            }
        else:
            return {
                'value': None,
                'confidence': 0.0,
                'method': 'none',
                'all_results': all_results
            }

    def detect_both(self, image: np.ndarray) -> Dict:
        """
        Detect both Class and Section values.

        Args:
            image: Input image

        Returns:
            Dictionary with both detection results
        """
        return {
            'class': self.detect(image, 'class'),
            'section': self.detect(image, 'section')
        }


def create_detector(vit_model_path: str = None, use_ocr: bool = True):
    """
    Factory function to create a RobustClassSectionDetector.

    Args:
        vit_model_path: Optional path to trained ViT model
        use_ocr: Whether to enable OCR fallback

    Returns:
        RobustClassSectionDetector instance
    """
    vit_model = None
    ocr_extractor = None

    # Try to load ViT model
    if vit_model_path and os.path.exists(vit_model_path):
        try:
            from src.ai.vit_bubble_classifier import ViTBubbleClassifierInference
            vit_model = ViTBubbleClassifierInference(vit_model_path)
            print(f"Loaded ViT model from {vit_model_path}")
        except Exception as e:
            print(f"Could not load ViT model: {e}")

    # Try to setup OCR
    if use_ocr:
        try:
            from src.ai.ocr_extractor import OCRExtractor
            ocr_extractor = OCRExtractor()
            print("OCR extractor initialized")
        except Exception as e:
            print(f"Could not initialize OCR: {e}")

    return RobustClassSectionDetector(vit_model=vit_model, ocr_extractor=ocr_extractor)


if __name__ == '__main__':
    # Test the detector
    import sys

    if len(sys.argv) < 2:
        print("Usage: python class_section_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image {image_path}")
        sys.exit(1)

    # Create detector (without ViT model for testing)
    detector = RobustClassSectionDetector()

    # Detect
    results = detector.detect_both(image)

    print("\nDetection Results:")
    print("=" * 50)

    for field, result in results.items():
        print(f"\n{field.upper()}:")
        print(f"  Value: {result['value']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Method: {result['method']}")
        print(f"  All results:")
        for method, data in result['all_results'].items():
            print(f"    {method}: {data['value']} ({data['confidence']:.2%})")

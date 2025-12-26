"""
Handwritten text recognition module.

Uses TrOCR or EasyOCR for recognizing handwritten text on OMR sheets.
Specifically optimized for:
- Block capital letters
- Name fields
- Numeric fields (roll numbers, school codes)
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextRecognitionResult:
    """Result of text recognition."""
    text: str
    confidence: float
    bounding_box: Optional[Tuple[int, int, int, int]] = None


class TextRecognizer:
    """
    Handwritten text recognizer for OMR sheets.

    Supports multiple backends:
    1. TrOCR (best accuracy, requires transformers)
    2. EasyOCR (good balance of speed/accuracy)
    3. Tesseract (fallback, less accurate for handwriting)
    """

    def __init__(
        self,
        backend: str = "easyocr",
        model_path: Optional[str] = None,
        use_gpu: bool = False
    ):
        """
        Initialize the text recognizer.

        Args:
            backend: Recognition backend ("trocr", "easyocr", "tesseract")
            model_path: Path to custom model (for TrOCR)
            use_gpu: Whether to use GPU acceleration
        """
        self.backend = backend
        self.use_gpu = use_gpu
        self.model = None
        self.processor = None

        self._init_backend(model_path)

    def _init_backend(self, model_path: Optional[str]):
        """Initialize the selected backend."""
        if self.backend == "trocr":
            self._init_trocr(model_path)
        elif self.backend == "easyocr":
            self._init_easyocr()
        else:
            self._init_tesseract()

    def _init_trocr(self, model_path: Optional[str]):
        """Initialize TrOCR model."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            model_name = model_path or "microsoft/trocr-base-handwritten"

            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()

            self.model.eval()
            print(f"TrOCR initialized: {model_name}")

        except ImportError:
            print("TrOCR not available, falling back to EasyOCR")
            self.backend = "easyocr"
            self._init_easyocr()

    def _init_easyocr(self):
        """Initialize EasyOCR."""
        try:
            import easyocr

            self.model = easyocr.Reader(
                ['en'],
                gpu=self.use_gpu,
                verbose=False
            )
            print("EasyOCR initialized")

        except ImportError:
            print("EasyOCR not available, falling back to Tesseract")
            self.backend = "tesseract"
            self._init_tesseract()

    def _init_tesseract(self):
        """Initialize Tesseract OCR."""
        try:
            import pytesseract
            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            self.model = pytesseract
            print("Tesseract initialized")
        except Exception:
            print("Warning: No OCR backend available")
            self.model = None

    def recognize(
        self,
        image: np.ndarray,
        region: Optional[Tuple[int, int, int, int]] = None,
        preprocess: bool = True
    ) -> TextRecognitionResult:
        """
        Recognize text in the image.

        Args:
            image: Input BGR image
            region: Optional (x, y, width, height) for text region
            preprocess: Whether to apply preprocessing

        Returns:
            TextRecognitionResult with recognized text
        """
        # Extract region if specified
        if region:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
        else:
            roi = image

        # Preprocess
        if preprocess:
            roi = self._preprocess(roi)

        # Recognize based on backend
        if self.backend == "trocr":
            return self._recognize_trocr(roi)
        elif self.backend == "easyocr":
            return self._recognize_easyocr(roi)
        else:
            return self._recognize_tesseract(roi)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Resize if too small
        h, w = gray.shape
        if h < 50:
            scale = 50 / h
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Denoise
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Binarize using adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        return binary

    def _recognize_trocr(self, image: np.ndarray) -> TextRecognitionResult:
        """Recognize text using TrOCR."""
        try:
            import torch
            from PIL import Image

            # Convert to RGB PIL Image
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pil_image = Image.fromarray(image)

            # Process
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values

            if self.use_gpu and torch.cuda.is_available():
                pixel_values = pixel_values.cuda()

            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)

            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return TextRecognitionResult(
                text=text.upper(),
                confidence=0.9  # TrOCR doesn't provide confidence scores easily
            )

        except Exception as e:
            return TextRecognitionResult(text="", confidence=0.0)

    def _recognize_easyocr(self, image: np.ndarray) -> TextRecognitionResult:
        """Recognize text using EasyOCR."""
        try:
            results = self.model.readtext(image, detail=1)

            if results:
                # Combine all detected text
                texts = []
                total_conf = 0

                for bbox, text, conf in results:
                    texts.append(text)
                    total_conf += conf

                combined_text = ' '.join(texts)
                avg_conf = total_conf / len(results) if results else 0

                return TextRecognitionResult(
                    text=combined_text.upper(),
                    confidence=avg_conf
                )
            else:
                return TextRecognitionResult(text="", confidence=0.0)

        except Exception as e:
            return TextRecognitionResult(text="", confidence=0.0)

    def _recognize_tesseract(self, image: np.ndarray) -> TextRecognitionResult:
        """Recognize text using Tesseract."""
        try:
            # Configure Tesseract for single line, uppercase
            config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '

            text = self.model.image_to_string(image, config=config)
            text = text.strip().upper()

            # Get confidence
            data = self.model.image_to_data(image, output_type=self.model.Output.DICT)
            confs = [int(c) for c in data['conf'] if int(c) > 0]
            avg_conf = sum(confs) / len(confs) / 100 if confs else 0

            return TextRecognitionResult(
                text=text,
                confidence=avg_conf
            )

        except Exception as e:
            return TextRecognitionResult(text="", confidence=0.0)

    def recognize_name_field(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> str:
        """
        Specialized recognition for handwritten name fields.

        Applies additional preprocessing optimized for block letters.
        """
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]

        # Additional preprocessing for names
        processed = self._preprocess_name_field(roi)

        result = self.recognize(processed, preprocess=False)

        # Post-process: remove non-alphabetic chars except spaces
        cleaned = ''.join(c for c in result.text if c.isalpha() or c.isspace())
        cleaned = ' '.join(cleaned.split())  # Normalize spaces

        return cleaned

    def _preprocess_name_field(self, image: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for name fields."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Remove grid lines using morphological operations
        # Horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)

        # Vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)

        # Remove lines
        cleaned = gray.copy()
        cleaned[horizontal_lines < 200] = 255
        cleaned[vertical_lines < 200] = 255

        # Enhance text
        cleaned = cv2.GaussianBlur(cleaned, (3, 3), 0)

        # Threshold
        _, binary = cv2.threshold(cleaned, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def recognize_number_field(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int],
        num_digits: int = 3
    ) -> str:
        """
        Specialized recognition for numeric fields.

        Args:
            image: Input image
            region: Region containing numbers
            num_digits: Expected number of digits

        Returns:
            Recognized digits as string
        """
        x, y, w, h = region
        roi = image[y:y+h, x:x+w]

        # Preprocess for numbers
        processed = self._preprocess(roi)

        result = self.recognize(processed, preprocess=False)

        # Keep only digits
        digits = ''.join(c for c in result.text if c.isdigit())

        # Pad or truncate to expected length
        if len(digits) < num_digits:
            digits = digits.zfill(num_digits)
        elif len(digits) > num_digits:
            digits = digits[:num_digits]

        return digits

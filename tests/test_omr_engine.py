"""
Tests for OMR Engine
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest


class TestOMREngine(unittest.TestCase):
    """Test OMR processing engine"""

    def setUp(self):
        """Set up test fixtures"""
        self.project_root = Path(__file__).parent.parent
        self.sample_blank = self.project_root / 'sampleimage' / 'omr.jpg'
        self.sample_filled = self.project_root / 'omrfilled' / 'omr-filled.png'

    def test_imports(self):
        """Test that all modules can be imported"""
        from src.core.omr_engine import OMREngine
        from src.core.template import TemplateLoader
        from src.core.bubble_detector import BubbleDetector
        from src.core.image_processor import ImageProcessor
        from src.models.candidate import Candidate
        from src.models.result import ProcessingResult
        from src.models.answer_key import AnswerKey
        from src.services.scorer import Scorer
        from src.services.batch_processor import BatchProcessor
        from src.services.excel_exporter import ExcelExporter

        self.assertTrue(True)

    def test_engine_initialization(self):
        """Test engine initializes correctly"""
        from src.core.omr_engine import OMREngine

        engine = OMREngine()
        self.assertIsNotNone(engine.template)
        self.assertIsNotNone(engine.bubble_detector)
        self.assertIsNotNone(engine.image_processor)

    def test_template_default(self):
        """Test default template creation"""
        from src.core.template import TemplateLoader

        loader = TemplateLoader()
        template = loader.create_default_template()

        self.assertEqual(template.name, "Vidya Bharti Pratibha Khoj 2025-26")
        self.assertEqual(template.page_dimensions, (2480, 3507))
        self.assertIn('CandidateName', template.field_blocks)
        self.assertIn('RollNumber', template.field_blocks)
        self.assertIn('Answers_1_10', template.field_blocks)

    def test_sample_image_exists(self):
        """Test sample images exist"""
        self.assertTrue(self.sample_blank.exists(), "Blank sample not found")
        self.assertTrue(self.sample_filled.exists(), "Filled sample not found")

    def test_process_filled_sample(self):
        """Test processing filled sample (if OpenCV is available)"""
        try:
            import cv2
            from src.core.omr_engine import OMREngine

            engine = OMREngine()
            result = engine.process_image(str(self.sample_filled))

            # Basic checks - actual accuracy depends on template calibration
            self.assertIsNotNone(result)
            # Note: Full accuracy requires template calibration

        except ImportError:
            self.skipTest("OpenCV not installed")

    def test_candidate_model(self):
        """Test Candidate model"""
        from src.models.candidate import Candidate

        candidate = Candidate()
        candidate.name = "TEST NAME"
        candidate.roll_number = "123"
        candidate.set_answer(1, "A")
        candidate.set_answer(2, "B")
        candidate.set_answer(3, "MULTI")

        self.assertEqual(candidate.name, "TEST NAME")
        self.assertEqual(candidate.get_answer(1), "A")
        self.assertEqual(candidate.multi_marked_count, 1)

    def test_answer_key_model(self):
        """Test AnswerKey model"""
        from src.models.answer_key import AnswerKey

        key = AnswerKey()
        key.set_answer(1, "A")
        key.set_answer(2, "B")

        self.assertEqual(key.get_answer(1), "A")
        self.assertEqual(key.get_filled_count(), 2)
        self.assertFalse(key.is_complete())

    def test_scorer(self):
        """Test Scorer service"""
        from src.models.candidate import Candidate
        from src.models.answer_key import AnswerKey
        from src.services.scorer import Scorer

        # Create answer key
        key = AnswerKey()
        for i in range(1, 61):
            key.set_answer(i, "A")

        # Create candidate
        candidate = Candidate()
        for i in range(1, 31):
            candidate.set_answer(i, "A")  # 30 correct
        for i in range(31, 51):
            candidate.set_answer(i, "B")  # 20 wrong
        # 10 unanswered

        # Score
        scorer = Scorer()
        scorer.set_answer_key(key)
        result = scorer.evaluate(candidate)

        self.assertEqual(result.correct_count, 30)
        self.assertEqual(result.wrong_count, 20)
        self.assertEqual(result.unanswered_count, 10)

        # Score: 30*1 + 20*(-0.25) + 10*0 = 30 - 5 = 25
        self.assertEqual(result.total_score, 25.0)


if __name__ == '__main__':
    unittest.main()

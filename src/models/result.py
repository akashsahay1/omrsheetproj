"""
Processing and evaluation result models
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from src.models.candidate import Candidate


class QuestionResult(Enum):
    """Result of individual question evaluation"""
    CORRECT = "correct"
    WRONG = "wrong"
    UNANSWERED = "unanswered"
    MULTI_MARKED = "multi_marked"


@dataclass
class ProcessingResult:
    """Result of processing a single OMR sheet"""

    # Processing status
    success: bool = False
    error_message: str = ""

    # Extracted data
    candidate: Optional[Candidate] = None

    # Processing metadata
    source_file: str = ""
    processing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Warnings and flags
    warnings: List[str] = field(default_factory=list)
    alignment_confidence: float = 0.0
    detection_confidence: float = 100.0  # Overall detection confidence (0-100%)
    needs_manual_review: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'error_message': self.error_message,
            'candidate': self.candidate.to_dict() if self.candidate else None,
            'source_file': self.source_file,
            'processing_time_ms': self.processing_time_ms,
            'timestamp': self.timestamp,
            'warnings': self.warnings,
            'alignment_confidence': self.alignment_confidence,
            'detection_confidence': self.detection_confidence,
            'needs_manual_review': self.needs_manual_review
        }


@dataclass
class EvaluationResult:
    """Result of evaluating a candidate's answers against answer key"""

    # Reference to candidate
    candidate: Optional[Candidate] = None

    # Scores
    total_score: float = 0.0
    max_possible_score: float = 60.0
    percentage: float = 0.0

    # Counts
    correct_count: int = 0
    wrong_count: int = 0
    unanswered_count: int = 0
    multi_marked_count: int = 0

    # Question-wise results
    question_results: Dict[int, QuestionResult] = field(default_factory=dict)

    # Rank (set after all evaluations)
    rank: int = 0

    def calculate_percentage(self) -> float:
        """Calculate percentage score"""
        if self.max_possible_score > 0:
            self.percentage = round((self.total_score / self.max_possible_score) * 100, 2)
        return self.percentage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'candidate': self.candidate.to_dict() if self.candidate else None,
            'total_score': self.total_score,
            'max_possible_score': self.max_possible_score,
            'percentage': self.percentage,
            'correct_count': self.correct_count,
            'wrong_count': self.wrong_count,
            'unanswered_count': self.unanswered_count,
            'multi_marked_count': self.multi_marked_count,
            'question_results': {k: v.value for k, v in self.question_results.items()},
            'rank': self.rank
        }


@dataclass
class BatchResult:
    """Result of batch processing multiple sheets"""

    # Overall status
    total_sheets: int = 0
    processed_count: int = 0
    success_count: int = 0
    failed_count: int = 0

    # Individual results
    results: List[ProcessingResult] = field(default_factory=list)

    # Evaluation results (after scoring)
    evaluation_results: List[EvaluationResult] = field(default_factory=list)

    # Timing
    start_time: str = ""
    end_time: str = ""
    total_time_seconds: float = 0.0

    # Output file path (where Excel was exported)
    output_path: str = ""

    # Statistics (calculated after processing)
    average_score: float = 0.0
    highest_score: float = 0.0
    lowest_score: float = 0.0
    score_distribution: Dict[str, int] = field(default_factory=dict)

    def add_result(self, result: ProcessingResult) -> None:
        """Add a processing result"""
        self.results.append(result)
        self.processed_count += 1
        if result.success:
            self.success_count += 1
        else:
            self.failed_count += 1

    def calculate_statistics(self) -> None:
        """Calculate batch statistics after all evaluations"""
        if not self.evaluation_results:
            return

        scores = [r.total_score for r in self.evaluation_results]

        if scores:
            self.average_score = round(sum(scores) / len(scores), 2)
            self.highest_score = max(scores)
            self.lowest_score = min(scores)

            # Score distribution (ranges)
            self.score_distribution = {
                '0-10': sum(1 for s in scores if 0 <= s < 10),
                '10-20': sum(1 for s in scores if 10 <= s < 20),
                '20-30': sum(1 for s in scores if 20 <= s < 30),
                '30-40': sum(1 for s in scores if 30 <= s < 40),
                '40-50': sum(1 for s in scores if 40 <= s < 50),
                '50-60': sum(1 for s in scores if 50 <= s <= 60)
            }

    def assign_ranks(self) -> None:
        """Assign ranks based on scores"""
        # Sort by score descending
        sorted_results = sorted(self.evaluation_results,
                                key=lambda x: x.total_score,
                                reverse=True)

        current_rank = 1
        previous_score = None

        for i, result in enumerate(sorted_results):
            if previous_score is not None and result.total_score < previous_score:
                current_rank = i + 1
            result.rank = current_rank
            previous_score = result.total_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_sheets': self.total_sheets,
            'processed_count': self.processed_count,
            'success_count': self.success_count,
            'failed_count': self.failed_count,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_time_seconds': self.total_time_seconds,
            'output_path': self.output_path,
            'average_score': self.average_score,
            'highest_score': self.highest_score,
            'lowest_score': self.lowest_score,
            'score_distribution': self.score_distribution,
            'results': [r.to_dict() for r in self.results],
            'evaluation_results': [r.to_dict() for r in self.evaluation_results]
        }

"""
Scoring and evaluation service
"""
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.models.candidate import Candidate
from src.models.answer_key import AnswerKey
from src.models.result import EvaluationResult, QuestionResult
from src.config.constants import ScoringConfig, MULTI_MARK_VALUE
from src.utils.logging_utils import get_logger

logger = get_logger('scorer')


@dataclass
class ScoringRules:
    """Scoring configuration"""
    correct_marks: float = ScoringConfig.CORRECT_MARKS
    wrong_marks: float = ScoringConfig.WRONG_MARKS
    unanswered_marks: float = ScoringConfig.UNANSWERED_MARKS
    multi_marked_marks: float = ScoringConfig.MULTI_MARKED_MARKS


class Scorer:
    """
    Scoring and evaluation service

    Evaluates candidate answers against answer key and calculates scores
    """

    def __init__(self, rules: Optional[ScoringRules] = None):
        """
        Initialize scorer

        Args:
            rules: Scoring rules (uses default if not provided)
        """
        self.rules = rules or ScoringRules()
        self.answer_key: Optional[AnswerKey] = None

    def set_answer_key(self, answer_key: Optional[AnswerKey]) -> None:
        """Set the answer key for evaluation (None clears it)"""
        self.answer_key = answer_key
        if answer_key is None:
            logger.info("Answer key cleared")
        else:
            logger.info(f"Answer key set: {answer_key.get_filled_count()}/60 answers defined")

    def load_answer_key(self, file_path: str) -> bool:
        """
        Load answer key from file

        Args:
            file_path: Path to answer key file (Excel/CSV/JSON)

        Returns:
            True if loaded successfully
        """
        try:
            self.answer_key = AnswerKey.from_file(file_path)
            logger.info(f"Loaded answer key from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load answer key: {e}")
            return False

    def set_manual_answers(self, answers: Dict[int, str]) -> None:
        """
        Set answer key manually

        Args:
            answers: Dict mapping question number to correct answer
        """
        self.answer_key = AnswerKey()
        for q_num, answer in answers.items():
            self.answer_key.set_answer(q_num, answer)

    def set_scoring_rules(self,
                          correct: float = 1.0,
                          wrong: float = -0.25,
                          unanswered: float = 0.0,
                          multi_marked: float = -0.25) -> None:
        """Set scoring rules"""
        self.rules = ScoringRules(
            correct_marks=correct,
            wrong_marks=wrong,
            unanswered_marks=unanswered,
            multi_marked_marks=multi_marked
        )

    def evaluate(self, candidate: Candidate) -> EvaluationResult:
        """
        Evaluate a single candidate's answers

        Args:
            candidate: Candidate with answers

        Returns:
            EvaluationResult with scores and breakdown
        """
        result = EvaluationResult(candidate=candidate)

        if self.answer_key is None:
            logger.warning("No answer key set, returning zero scores")
            return result

        total_score = 0.0
        correct_count = 0
        wrong_count = 0
        unanswered_count = 0
        multi_marked_count = 0

        for q_num in range(1, 61):
            candidate_answer = candidate.get_answer(q_num)
            correct_answer = self.answer_key.get_answer(q_num)

            # Determine result for this question
            if candidate_answer == MULTI_MARK_VALUE:
                # Multi-marked
                result.question_results[q_num] = QuestionResult.MULTI_MARKED
                total_score += self.rules.multi_marked_marks
                multi_marked_count += 1

            elif candidate_answer == "" or candidate_answer is None:
                # Unanswered
                result.question_results[q_num] = QuestionResult.UNANSWERED
                total_score += self.rules.unanswered_marks
                unanswered_count += 1

            elif candidate_answer == correct_answer:
                # Correct
                result.question_results[q_num] = QuestionResult.CORRECT
                total_score += self.rules.correct_marks
                correct_count += 1

            else:
                # Wrong
                result.question_results[q_num] = QuestionResult.WRONG
                total_score += self.rules.wrong_marks
                wrong_count += 1

        # Set counts
        result.total_score = round(total_score, 2)
        result.correct_count = correct_count
        result.wrong_count = wrong_count
        result.unanswered_count = unanswered_count
        result.multi_marked_count = multi_marked_count

        # Calculate percentage
        result.calculate_percentage()

        logger.debug(f"Evaluated {candidate.name}: score={result.total_score}, "
                    f"correct={correct_count}, wrong={wrong_count}")

        return result

    def evaluate_batch(self, candidates: List[Candidate]) -> List[EvaluationResult]:
        """
        Evaluate multiple candidates

        Args:
            candidates: List of candidates

        Returns:
            List of EvaluationResult objects with ranks assigned
        """
        results = []

        for candidate in candidates:
            result = self.evaluate(candidate)
            results.append(result)

        # Assign ranks
        self._assign_ranks(results)

        return results

    def _assign_ranks(self, results: List[EvaluationResult]) -> None:
        """Assign ranks based on scores"""
        # Sort by score descending
        sorted_results = sorted(results, key=lambda r: r.total_score, reverse=True)

        current_rank = 1
        previous_score = None

        for i, result in enumerate(sorted_results):
            if previous_score is not None and result.total_score < previous_score:
                current_rank = i + 1
            result.rank = current_rank
            previous_score = result.total_score

    def get_statistics(self, results: List[EvaluationResult]) -> Dict:
        """
        Calculate statistics for a batch of results

        Args:
            results: List of EvaluationResult

        Returns:
            Dict with statistics
        """
        if not results:
            return {
                'count': 0,
                'average_score': 0,
                'highest_score': 0,
                'lowest_score': 0,
                'average_correct': 0,
                'average_wrong': 0,
                'score_distribution': {}
            }

        scores = [r.total_score for r in results]
        correct_counts = [r.correct_count for r in results]
        wrong_counts = [r.wrong_count for r in results]

        # Score distribution
        distribution = {
            '0-10': sum(1 for s in scores if 0 <= s < 10),
            '10-20': sum(1 for s in scores if 10 <= s < 20),
            '20-30': sum(1 for s in scores if 20 <= s < 30),
            '30-40': sum(1 for s in scores if 30 <= s < 40),
            '40-50': sum(1 for s in scores if 40 <= s < 50),
            '50-60': sum(1 for s in scores if 50 <= s <= 60),
            'negative': sum(1 for s in scores if s < 0)
        }

        return {
            'count': len(results),
            'average_score': round(sum(scores) / len(scores), 2),
            'highest_score': max(scores),
            'lowest_score': min(scores),
            'average_correct': round(sum(correct_counts) / len(correct_counts), 1),
            'average_wrong': round(sum(wrong_counts) / len(wrong_counts), 1),
            'score_distribution': distribution
        }

    def get_question_analysis(self, results: List[EvaluationResult]) -> Dict[int, Dict]:
        """
        Analyze performance on each question

        Args:
            results: List of EvaluationResult

        Returns:
            Dict mapping question number to analysis
        """
        analysis = {}

        for q_num in range(1, 61):
            correct = 0
            wrong = 0
            unanswered = 0
            multi_marked = 0

            for result in results:
                q_result = result.question_results.get(q_num)

                if q_result == QuestionResult.CORRECT:
                    correct += 1
                elif q_result == QuestionResult.WRONG:
                    wrong += 1
                elif q_result == QuestionResult.UNANSWERED:
                    unanswered += 1
                elif q_result == QuestionResult.MULTI_MARKED:
                    multi_marked += 1

            total = len(results)
            analysis[q_num] = {
                'correct': correct,
                'wrong': wrong,
                'unanswered': unanswered,
                'multi_marked': multi_marked,
                'correct_percentage': round((correct / total) * 100, 1) if total > 0 else 0,
                'difficulty': 'hard' if (correct / total < 0.3) else ('medium' if correct / total < 0.7 else 'easy')
            }

        return analysis

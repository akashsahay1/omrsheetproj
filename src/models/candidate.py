"""
Candidate data model
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Candidate:
    """Represents a candidate's extracted information from OMR sheet"""

    # Personal Information
    name: str = ""
    roll_number: str = ""
    class_: str = ""  # 'class' is reserved keyword
    section: str = ""
    school_code: str = ""

    # Answers (1-60)
    answers: Dict[int, str] = field(default_factory=dict)

    # Flags
    multi_marked_questions: List[int] = field(default_factory=list)
    unanswered_questions: List[int] = field(default_factory=list)

    # Source information
    source_image: str = ""
    image_index: int = 0

    # Detection confidence (0-100%)
    detection_confidence: float = 100.0

    def __post_init__(self):
        """Initialize answers dict if empty"""
        if not self.answers:
            self.answers = {i: "" for i in range(1, 61)}

    def get_answer(self, question_number: int) -> str:
        """Get answer for a specific question"""
        return self.answers.get(question_number, "")

    def set_answer(self, question_number: int, answer: str) -> None:
        """Set answer for a specific question"""
        if 1 <= question_number <= 60:
            self.answers[question_number] = answer

            # Update flags
            if answer == "MULTI" and question_number not in self.multi_marked_questions:
                self.multi_marked_questions.append(question_number)
            elif answer == "" and question_number not in self.unanswered_questions:
                self.unanswered_questions.append(question_number)

    @property
    def answered_count(self) -> int:
        """Count of answered questions"""
        return sum(1 for a in self.answers.values() if a and a != "MULTI")

    @property
    def multi_marked_count(self) -> int:
        """Count of multi-marked questions"""
        return len(self.multi_marked_questions)

    @property
    def unanswered_count(self) -> int:
        """Count of unanswered questions"""
        return 60 - self.answered_count - self.multi_marked_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'roll_number': self.roll_number,
            'class': self.class_,
            'section': self.section,
            'school_code': self.school_code,
            'answers': self.answers,
            'multi_marked_questions': self.multi_marked_questions,
            'unanswered_questions': self.unanswered_questions,
            'source_image': self.source_image,
            'answered_count': self.answered_count,
            'multi_marked_count': self.multi_marked_count,
            'unanswered_count': self.unanswered_count,
            'detection_confidence': self.detection_confidence
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Candidate':
        """Create from dictionary"""
        return cls(
            name=data.get('name', ''),
            roll_number=data.get('roll_number', ''),
            class_=data.get('class', ''),
            section=data.get('section', ''),
            school_code=data.get('school_code', ''),
            answers=data.get('answers', {}),
            multi_marked_questions=data.get('multi_marked_questions', []),
            unanswered_questions=data.get('unanswered_questions', []),
            source_image=data.get('source_image', ''),
            image_index=data.get('image_index', 0),
            detection_confidence=data.get('detection_confidence', 100.0)
        )

    def __str__(self) -> str:
        """String representation"""
        return f"Candidate(name='{self.name}', roll='{self.roll_number}', class='{self.class_}')"

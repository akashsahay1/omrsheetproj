"""
Answer key model
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import json
import csv
from pathlib import Path


@dataclass
class AnswerKey:
    """Represents the correct answers for an exam"""

    # Answer key: question_number -> correct_answer
    answers: Dict[int, str] = field(default_factory=dict)

    # Metadata
    name: str = ""
    description: str = ""
    total_questions: int = 60

    def __post_init__(self):
        """Initialize with empty answers if not provided"""
        if not self.answers:
            self.answers = {i: "" for i in range(1, self.total_questions + 1)}

    def set_answer(self, question_number: int, answer: str) -> bool:
        """Set answer for a question"""
        if 1 <= question_number <= self.total_questions:
            answer = answer.upper().strip()
            if answer in ['A', 'B', 'C', 'D', '']:
                self.answers[question_number] = answer
                return True
        return False

    def get_answer(self, question_number: int) -> str:
        """Get answer for a question"""
        return self.answers.get(question_number, "")

    def is_complete(self) -> bool:
        """Check if all answers are filled"""
        return all(self.answers.get(i, "") != "" for i in range(1, self.total_questions + 1))

    def get_filled_count(self) -> int:
        """Get count of filled answers"""
        return sum(1 for a in self.answers.values() if a != "")

    def validate(self) -> List[str]:
        """Validate answer key and return list of issues"""
        issues = []

        for q_num, answer in self.answers.items():
            if answer and answer not in ['A', 'B', 'C', 'D']:
                issues.append(f"Question {q_num}: Invalid answer '{answer}'")

        missing = [q for q in range(1, self.total_questions + 1)
                   if self.answers.get(q, "") == ""]
        if missing:
            issues.append(f"Missing answers for questions: {missing}")

        return issues

    @classmethod
    def from_excel(cls, file_path: str) -> 'AnswerKey':
        """Load answer key from Excel file"""
        try:
            from openpyxl import load_workbook

            wb = load_workbook(file_path)
            ws = wb.active

            answer_key = cls()

            # Try to find question number and answer columns
            # Expected format: Column A = Question Number, Column B = Answer
            # Or just 60 rows with answers in column A/B

            for row_idx, row in enumerate(ws.iter_rows(min_row=1, max_row=61, values_only=True), 1):
                if row_idx == 1:
                    # Check if header row
                    first_cell = str(row[0]).lower() if row[0] else ""
                    if 'question' in first_cell or 'q.' in first_cell or first_cell == 'q':
                        continue  # Skip header

                # Try to parse question number and answer
                if len(row) >= 2 and row[0] is not None:
                    try:
                        q_num = int(row[0])
                        answer = str(row[1]).upper().strip() if row[1] else ""
                        if 1 <= q_num <= 60 and answer in ['A', 'B', 'C', 'D', '']:
                            answer_key.set_answer(q_num, answer)
                    except (ValueError, TypeError):
                        # Try just the answer in column A
                        answer = str(row[0]).upper().strip() if row[0] else ""
                        if answer in ['A', 'B', 'C', 'D']:
                            answer_key.set_answer(row_idx, answer)
                elif row[0] is not None:
                    # Single column format
                    answer = str(row[0]).upper().strip()
                    if answer in ['A', 'B', 'C', 'D']:
                        answer_key.set_answer(row_idx, answer)

            answer_key.name = Path(file_path).stem
            return answer_key

        except Exception as e:
            raise ValueError(f"Failed to load Excel answer key: {e}")

    @classmethod
    def from_csv(cls, file_path: str) -> 'AnswerKey':
        """Load answer key from CSV file"""
        try:
            answer_key = cls()

            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)

                for row_idx, row in enumerate(reader, 1):
                    if not row:
                        continue

                    # Skip header row
                    if row_idx == 1:
                        first_cell = row[0].lower()
                        if 'question' in first_cell or first_cell in ['q', 'q.', 'no', 'no.']:
                            continue

                    # Parse row
                    if len(row) >= 2:
                        try:
                            q_num = int(row[0])
                            answer = row[1].upper().strip()
                            if 1 <= q_num <= 60 and answer in ['A', 'B', 'C', 'D', '']:
                                answer_key.set_answer(q_num, answer)
                        except ValueError:
                            answer = row[0].upper().strip()
                            if answer in ['A', 'B', 'C', 'D']:
                                answer_key.set_answer(row_idx, answer)
                    elif len(row) == 1:
                        answer = row[0].upper().strip()
                        if answer in ['A', 'B', 'C', 'D']:
                            answer_key.set_answer(row_idx, answer)

            answer_key.name = Path(file_path).stem
            return answer_key

        except Exception as e:
            raise ValueError(f"Failed to load CSV answer key: {e}")

    @classmethod
    def from_file(cls, file_path: str) -> 'AnswerKey':
        """Load answer key from file (auto-detect format)"""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in ['.xlsx', '.xls']:
            return cls.from_excel(file_path)
        elif ext == '.csv':
            return cls.from_csv(file_path)
        elif ext == '.json':
            return cls.from_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    @classmethod
    def from_json(cls, file_path: str) -> 'AnswerKey':
        """Load answer key from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            answer_key = cls()
            answer_key.name = data.get('name', Path(file_path).stem)
            answer_key.description = data.get('description', '')

            answers = data.get('answers', {})
            for q_num, answer in answers.items():
                answer_key.set_answer(int(q_num), answer)

            return answer_key

        except Exception as e:
            raise ValueError(f"Failed to load JSON answer key: {e}")

    def to_json(self, file_path: str) -> bool:
        """Save answer key to JSON file"""
        try:
            data = {
                'name': self.name,
                'description': self.description,
                'total_questions': self.total_questions,
                'answers': {str(k): v for k, v in self.answers.items()}
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True

        except Exception as e:
            print(f"Failed to save answer key: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'description': self.description,
            'total_questions': self.total_questions,
            'answers': self.answers,
            'filled_count': self.get_filled_count(),
            'is_complete': self.is_complete()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnswerKey':
        """Create from dictionary"""
        answer_key = cls()
        answer_key.name = data.get('name', '')
        answer_key.description = data.get('description', '')

        answers = data.get('answers', {})
        for q_num, answer in answers.items():
            answer_key.set_answer(int(q_num), answer)

        return answer_key

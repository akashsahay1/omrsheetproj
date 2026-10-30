"""
Answer key store

Keeps any number of named answer keys (one per exam) and remembers which
one is selected for scoring. Stored as JSON in the app data directory next
to settings.json.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.models.answer_key import AnswerKey
from src.utils.logging_utils import get_logger

logger = get_logger('answer_key_store')

TOTAL_QUESTIONS = 60
VALID_ANSWERS = {'A', 'B', 'C', 'D'}


def _clean_answers(answers: Optional[Dict[Any, Any]]) -> Dict[int, str]:
    """
    Normalise answers to {question_number(int): 'A'-'D'}.

    JSON (and the JS bridge) turns question numbers into strings; scoring
    looks answers up by int, so convert here.
    """
    cleaned: Dict[int, str] = {}
    for q, a in (answers or {}).items():
        try:
            q_num = int(q)
        except (TypeError, ValueError):
            continue
        a = str(a or '').strip().upper()
        if 1 <= q_num <= TOTAL_QUESTIONS and a in VALID_ANSWERS:
            cleaned[q_num] = a
    return cleaned


class AnswerKeyStore:
    """Persistent collection of named answer keys."""

    def __init__(self, storage_path: Optional[str] = None, legacy_key: Optional[Dict[str, Any]] = None):
        """
        Args:
            storage_path: JSON file path (default: ~/.omr_processor/answer_keys.json)
            legacy_key: the single answer key saved by older versions in
                settings.json; imported once if the store doesn't exist yet
        """
        if storage_path is None:
            app_data = Path.home() / ".omr_processor"
            app_data.mkdir(exist_ok=True)
            storage_path = str(app_data / "answer_keys.json")
        self._path = Path(storage_path)
        self._keys: List[Dict[str, Any]] = []
        self._active_id: Optional[str] = None
        self._load(legacy_key)

    # ------------------------------------------------------------ storage

    def _load(self, legacy_key: Optional[Dict[str, Any]]) -> None:
        if self._path.exists():
            try:
                with open(self._path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for item in data.get('keys', []):
                    if item.get('id') and str(item.get('name', '')).strip():
                        self._keys.append({
                            'id': item['id'],
                            'name': str(item['name']).strip(),
                            'answers': _clean_answers(item.get('answers')),
                            'updated': item.get('updated', ''),
                        })
                if any(k['id'] == data.get('active_id') for k in self._keys):
                    self._active_id = data['active_id']
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Could not load answer keys: {e}")
            return

        # First run of this version: carry over the old single saved key
        if legacy_key and _clean_answers(legacy_key.get('answers')):
            key = self._new_entry(legacy_key.get('name') or 'Saved Answer Key',
                                  legacy_key.get('answers'))
            self._keys.append(key)
            self._active_id = key['id']
            self._save()

    def _save(self) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'active_id': self._active_id,
                'keys': [{
                    'id': k['id'],
                    'name': k['name'],
                    'updated': k['updated'],
                    'answers': {str(q): a for q, a in sorted(k['answers'].items())},
                } for k in self._keys],
            }
            with open(self._path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            logger.error(f"Could not save answer keys: {e}")
            return False

    @staticmethod
    def _new_entry(name: str, answers=None) -> Dict[str, Any]:
        return {
            'id': uuid.uuid4().hex,
            'name': name.strip(),
            'answers': _clean_answers(answers),
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        }

    def _find(self, key_id: str) -> Optional[Dict[str, Any]]:
        return next((k for k in self._keys if k['id'] == key_id), None)

    def _name_taken(self, name: str, except_id: Optional[str] = None) -> bool:
        lowered = name.strip().lower()
        return any(k['name'].lower() == lowered and k['id'] != except_id for k in self._keys)

    # ------------------------------------------------------------ queries

    @staticmethod
    def summary(key: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'id': key['id'],
            'name': key['name'],
            'filled_count': len(key['answers']),
            'updated': key['updated'],
        }

    def list(self) -> List[Dict[str, Any]]:
        """Summaries of all keys, sorted by name."""
        return [self.summary(k) for k in sorted(self._keys, key=lambda k: k['name'].lower())]

    def get(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Full key (summary + answers as {"1": "A", ...}), or None."""
        key = self._find(key_id)
        if key is None:
            return None
        return {**self.summary(key), 'answers': {str(q): a for q, a in key['answers'].items()}}

    def to_answer_key(self, key_id: str) -> Optional[AnswerKey]:
        """AnswerKey model for scoring, or None."""
        key = self._find(key_id)
        if key is None:
            return None
        answers = {q: key['answers'].get(q, '') for q in range(1, TOTAL_QUESTIONS + 1)}
        return AnswerKey(answers=answers, name=key['name'], total_questions=TOTAL_QUESTIONS)

    @property
    def active_id(self) -> Optional[str]:
        return self._active_id

    # ------------------------------------------------------------ changes

    def create(self, name: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Create an empty key. Returns (summary, message)."""
        name = (name or '').strip()
        if not name:
            return None, 'Please enter an answer key name'
        if self._name_taken(name):
            return None, f'An answer key named "{name}" already exists'
        key = self._new_entry(name)
        self._keys.append(key)
        if not self._save():
            return None, 'Could not save answer keys'
        return self.summary(key), 'Created'

    def update(self, key_id: str, name: str, answers: Dict[Any, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
        """Rename a key and replace its answers. Returns (summary, message)."""
        key = self._find(key_id)
        if key is None:
            return None, 'Answer key not found'
        name = (name or '').strip()
        if not name:
            return None, 'Please enter an answer key name'
        if self._name_taken(name, except_id=key_id):
            return None, f'An answer key named "{name}" already exists'
        key['name'] = name
        key['answers'] = _clean_answers(answers)
        key['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        if not self._save():
            return None, 'Could not save answer keys'
        return self.summary(key), 'Saved'

    def delete(self, key_id: str) -> Tuple[bool, str]:
        key = self._find(key_id)
        if key is None:
            return False, 'Answer key not found'
        self._keys.remove(key)
        if self._active_id == key_id:
            self._active_id = None
        if not self._save():
            return False, 'Could not save answer keys'
        return True, 'Deleted'

    def set_active(self, key_id: Optional[str]) -> bool:
        """Select the key used for scoring (None to clear)."""
        if key_id and self._find(key_id) is None:
            return False
        self._active_id = key_id or None
        return self._save()

"""
School code registry

Maps school codes (as bubbled on the OMR sheet) to school names, so the
result Excel can show the school name instead of the client typing it in.

The list is stored as JSON in the app data directory next to settings.json.
On first use it is seeded from the bundled templates/school_codes.json.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openpyxl import load_workbook

from src.utils.logging_utils import get_logger

logger = get_logger('school_registry')


def _bundled_defaults_path() -> Path:
    """Default school list shipped with the app (works for dev and bundled app)."""
    if getattr(sys, 'frozen', False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).parent.parent.parent
    return base / 'templates' / 'school_codes.json'


def normalize_code(code) -> Optional[str]:
    """
    Normalise a school code for matching.

    The OMR reads three bubbled digits ("011") while the client's list uses
    plain numbers ("11"), so codes are compared as integers.
    Returns None for empty / non-numeric codes.
    """
    if code is None:
        return None
    text = str(code).strip()
    if text.endswith('.0'):  # numbers read from Excel
        text = text[:-2]
    if not text.isdigit():
        return None
    return str(int(text))


class SchoolRegistry:
    """Persistent list of school codes and names."""

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            app_data = Path.home() / ".omr_processor"
            app_data.mkdir(exist_ok=True)
            storage_path = str(app_data / "school_codes.json")
        self._path = Path(storage_path)
        self._schools: Dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------ storage

    def _load(self) -> None:
        source = self._path if self._path.exists() else _bundled_defaults_path()
        if not source.exists():
            return
        try:
            with open(source, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for item in data.get('schools', []):
                code = normalize_code(item.get('code'))
                name = str(item.get('name', '')).strip()
                if code and name:
                    self._schools[code] = name
            if source != self._path:
                self._save()  # first run: keep a user-editable copy
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Could not load school codes from {source}: {e}")

    def _save(self) -> bool:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, 'w', encoding='utf-8') as f:
                json.dump({'schools': self.list()}, f, indent=2, ensure_ascii=False)
            return True
        except IOError as e:
            logger.error(f"Could not save school codes: {e}")
            return False

    # ------------------------------------------------------------ queries

    def list(self) -> List[Dict[str, str]]:
        """All schools sorted by numeric code."""
        return [{'code': c, 'name': self._schools[c]}
                for c in sorted(self._schools, key=int)]

    def get_name(self, code) -> str:
        """School name for an OMR school code, or '' if unknown."""
        key = normalize_code(code)
        return self._schools.get(key, '') if key else ''

    # ------------------------------------------------------------ changes

    def save_school(self, code, name: str, original_code=None) -> Tuple[bool, str]:
        """
        Add a school, or update one. When editing, original_code is the code
        before the edit (so the code itself can be changed).
        """
        key = normalize_code(code)
        name = (name or '').strip()
        if not key:
            return False, 'School code must be a number'
        if not name:
            return False, 'School name is required'

        original = normalize_code(original_code) if original_code not in (None, '') else None
        if key in self._schools and key != original:
            return False, f'School code {key} already exists ({self._schools[key]})'

        if original and original != key:
            self._schools.pop(original, None)
        self._schools[key] = name
        if not self._save():
            return False, 'Could not save school codes'
        return True, 'Saved'

    def delete_school(self, code) -> Tuple[bool, str]:
        key = normalize_code(code)
        if not key or key not in self._schools:
            return False, 'School code not found'
        del self._schools[key]
        if not self._save():
            return False, 'Could not save school codes'
        return True, 'Deleted'

    def import_excel(self, file_path: str) -> Tuple[int, int, List[str]]:
        """
        Import schools from an Excel sheet with "Code" and "School ... Name"
        header columns (the header row may be below a title row).
        Existing codes are updated, new ones added.

        Returns (added, updated, skipped_row_messages).
        """
        wb = load_workbook(file_path, read_only=True, data_only=True)
        ws = wb.worksheets[0]
        rows = list(ws.iter_rows(values_only=True))
        wb.close()

        code_col = name_col = header_idx = None
        for idx, row in enumerate(rows[:20]):
            cells = [str(v).strip().lower() if v is not None else '' for v in row]
            codes = [i for i, v in enumerate(cells) if v == 'code' or v == 'school code']
            names = [i for i, v in enumerate(cells) if 'name' in v]
            if codes and names:
                code_col, name_col, header_idx = codes[0], names[0], idx
                break
        if header_idx is None:
            raise ValueError('Could not find "Code" and "School Name" columns in the sheet')

        added = updated = 0
        skipped: List[str] = []
        for line_no, row in enumerate(rows[header_idx + 1:], header_idx + 2):
            if row is None or all(v is None or str(v).strip() == '' for v in row):
                continue
            raw_code = row[code_col] if code_col < len(row) else None
            raw_name = row[name_col] if name_col < len(row) else None
            key = normalize_code(raw_code)
            name = str(raw_name).strip() if raw_name is not None else ''
            if not key or not name:
                skipped.append(f'Row {line_no}: missing or invalid code/name')
                continue
            if key in self._schools:
                if self._schools[key] != name:
                    updated += 1
            else:
                added += 1
            self._schools[key] = name

        self._save()
        return added, updated, skipped

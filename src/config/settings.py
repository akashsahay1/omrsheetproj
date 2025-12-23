"""
Application settings management
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ScoringSettings:
    """Scoring configuration"""
    correct_marks: float = 1.0
    wrong_marks: float = -0.25
    unanswered_marks: float = 0.0
    multi_marked_as_wrong: bool = True


@dataclass
class ProcessingSettings:
    """Processing configuration"""
    auto_align: bool = True
    debug_mode: bool = False
    save_debug_images: bool = False
    fill_threshold: int = 180
    min_fill_ratio: float = 0.3


@dataclass
class ExportSettings:
    """Export configuration"""
    include_summary: bool = True
    include_question_details: bool = True
    auto_calculate_ranks: bool = True
    default_export_path: str = ""


@dataclass
class Settings:
    """Main application settings"""
    scoring: ScoringSettings = field(default_factory=ScoringSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    export: ExportSettings = field(default_factory=ExportSettings)
    last_input_folder: str = ""
    last_output_folder: str = ""
    last_answer_key_path: str = ""
    window_width: int = 1280
    window_height: int = 800

    _settings_file: str = field(default="settings.json", repr=False)

    @classmethod
    def load(cls, settings_path: Optional[str] = None) -> 'Settings':
        """Load settings from file"""
        if settings_path is None:
            # Default to app data directory
            app_data = Path.home() / ".omr_processor"
            app_data.mkdir(exist_ok=True)
            settings_path = str(app_data / "settings.json")

        settings = cls()
        settings._settings_file = settings_path

        if os.path.exists(settings_path):
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    settings._load_from_dict(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load settings: {e}")

        return settings

    def _load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load settings from dictionary"""
        if 'scoring' in data:
            self.scoring = ScoringSettings(**data['scoring'])
        if 'processing' in data:
            self.processing = ProcessingSettings(**data['processing'])
        if 'export' in data:
            self.export = ExportSettings(**data['export'])

        # Load simple fields
        for key in ['last_input_folder', 'last_output_folder', 'last_answer_key_path',
                    'window_width', 'window_height']:
            if key in data:
                setattr(self, key, data[key])

    def save(self) -> bool:
        """Save settings to file"""
        try:
            data = {
                'scoring': asdict(self.scoring),
                'processing': asdict(self.processing),
                'export': asdict(self.export),
                'last_input_folder': self.last_input_folder,
                'last_output_folder': self.last_output_folder,
                'last_answer_key_path': self.last_answer_key_path,
                'window_width': self.window_width,
                'window_height': self.window_height
            }

            # Ensure directory exists
            Path(self._settings_file).parent.mkdir(parents=True, exist_ok=True)

            with open(self._settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        except IOError as e:
            print(f"Error saving settings: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'scoring': asdict(self.scoring),
            'processing': asdict(self.processing),
            'export': asdict(self.export),
            'last_input_folder': self.last_input_folder,
            'last_output_folder': self.last_output_folder,
            'last_answer_key_path': self.last_answer_key_path,
            'window_width': self.window_width,
            'window_height': self.window_height
        }

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update settings from dictionary"""
        self._load_from_dict(data)
        self.save()

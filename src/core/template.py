"""
Template configuration loader and handler
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from src.utils.logging_utils import get_logger

logger = get_logger('template')


@dataclass
class FieldBlock:
    """Represents a field block on the OMR sheet"""

    name: str = ""
    field_type: str = ""  # QTYPE_MCQ4, QTYPE_INT, QTYPE_ALPHA
    origin: Tuple[int, int] = (0, 0)  # Top-left corner (x, y)
    field_labels: List[str] = field(default_factory=list)
    bubble_values: List[str] = field(default_factory=list)
    bubbles_gap: int = 30  # Gap between bubbles
    labels_gap: int = 50   # Gap between labels/columns
    direction: str = "vertical"  # vertical or horizontal
    bubble_width: int = 32
    bubble_height: int = 32

    def get_bubble_coordinates(self) -> List[Dict[str, Any]]:
        """
        Calculate coordinates for all bubbles in this field block

        Returns:
            List of dicts with label, value, x, y, width, height
        """
        coordinates = []
        x_origin, y_origin = self.origin

        for label_idx, label in enumerate(self.field_labels):
            for value_idx, value in enumerate(self.bubble_values):
                if self.direction == "vertical":
                    # Bubbles arranged vertically, labels horizontally
                    x = x_origin + (label_idx * self.labels_gap)
                    y = y_origin + (value_idx * self.bubbles_gap)
                else:
                    # Bubbles arranged horizontally, labels vertically
                    x = x_origin + (value_idx * self.bubbles_gap)
                    y = y_origin + (label_idx * self.labels_gap)

                coordinates.append({
                    'label': label,
                    'value': value,
                    'x': x,
                    'y': y,
                    'width': self.bubble_width,
                    'height': self.bubble_height
                })

        return coordinates


@dataclass
class Template:
    """OMR sheet template configuration"""

    name: str = ""
    page_dimensions: Tuple[int, int] = (2480, 3507)
    bubble_dimensions: Tuple[int, int] = (32, 32)
    empty_value: str = ""
    multi_mark_value: str = "MULTI"

    # Field blocks
    field_blocks: Dict[str, FieldBlock] = field(default_factory=dict)

    # Preprocessing options
    preprocessors: List[Dict[str, Any]] = field(default_factory=list)

    def get_field_block(self, name: str) -> Optional[FieldBlock]:
        """Get field block by name"""
        return self.field_blocks.get(name)

    def get_all_bubble_coordinates(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all bubble coordinates organized by field block"""
        all_coords = {}
        for name, block in self.field_blocks.items():
            all_coords[name] = block.get_bubble_coordinates()
        return all_coords


class TemplateLoader:
    """Loads and manages OMR templates"""

    def __init__(self, templates_dir: Optional[str] = None):
        if templates_dir is None:
            # Use templates directory relative to project root
            # Try multiple possible locations
            possible_paths = [
                Path.cwd() / "templates",  # Current working directory
                Path(__file__).parent.parent.parent / "templates",  # Relative to this file
            ]
            self.templates_dir = None
            for p in possible_paths:
                if p.exists():
                    self.templates_dir = p
                    break
            if self.templates_dir is None:
                self.templates_dir = Path.cwd() / "templates"
        else:
            self.templates_dir = Path(templates_dir)

        self._templates_cache: Dict[str, Template] = {}
        logger.debug(f"Templates directory: {self.templates_dir}")

    def load_template(self, template_name: str) -> Optional[Template]:
        """
        Load template from JSON file

        Args:
            template_name: Name of template (folder name)

        Returns:
            Template object or None if failed
        """
        # Check cache
        if template_name in self._templates_cache:
            return self._templates_cache[template_name]

        # Find template file
        template_path = self.templates_dir / template_name / "template.json"

        if not template_path.exists():
            logger.error(f"Template not found: {template_path}")
            return None

        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            template = self._parse_template(data)
            self._templates_cache[template_name] = template

            logger.info(f"Loaded template: {template_name}")
            return template

        except Exception as e:
            logger.error(f"Failed to load template {template_name}: {e}")
            return None

    def _parse_template(self, data: Dict[str, Any]) -> Template:
        """Parse template from JSON data"""
        template = Template()

        template.name = data.get('templateName', '')
        template.page_dimensions = tuple(data.get('pageDimensions', [2480, 3507]))
        template.bubble_dimensions = tuple(data.get('bubbleDimensions', [32, 32]))
        template.empty_value = data.get('emptyValue', '')
        template.multi_mark_value = data.get('multiMarkValue', 'MULTI')
        template.preprocessors = data.get('preProcessors', [])

        # Parse field blocks
        field_blocks_data = data.get('fieldBlocks', {})
        for name, block_data in field_blocks_data.items():
            block = FieldBlock(
                name=name,
                field_type=block_data.get('fieldType', ''),
                origin=tuple(block_data.get('origin', [0, 0])),
                field_labels=block_data.get('fieldLabels', []),
                bubble_values=block_data.get('bubbleValues', []),
                bubbles_gap=block_data.get('bubblesGap', 30),
                labels_gap=block_data.get('labelsGap', 50),
                direction=block_data.get('direction', 'vertical'),
                bubble_width=template.bubble_dimensions[0],
                bubble_height=template.bubble_dimensions[1]
            )
            template.field_blocks[name] = block

        return template

    def get_available_templates(self) -> List[str]:
        """Get list of available templates"""
        templates = []

        if not self.templates_dir.exists():
            return templates

        for item in self.templates_dir.iterdir():
            if item.is_dir():
                template_file = item / "template.json"
                if template_file.exists():
                    templates.append(item.name)

        return templates

    def create_default_template(self) -> Template:
        """
        Create default template for Vidya Bharti OMR sheet

        This is used when no template file exists
        """
        template = Template()
        template.name = "Vidya Bharti Pratibha Khoj 2025-26"
        template.page_dimensions = (2480, 3507)
        template.bubble_dimensions = (32, 32)

        # Name grid (25 columns x 26 rows A-Z)
        template.field_blocks['CandidateName'] = FieldBlock(
            name='CandidateName',
            field_type='QTYPE_ALPHA',
            origin=(85, 180),
            field_labels=[f'Name_{i}' for i in range(1, 26)],
            bubble_values=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            bubbles_gap=28,
            labels_gap=38,
            direction='vertical',
            bubble_width=32,
            bubble_height=32
        )

        # Roll Number (3 digits)
        template.field_blocks['RollNumber'] = FieldBlock(
            name='RollNumber',
            field_type='QTYPE_INT',
            origin=(1650, 205),
            field_labels=['Roll_D1', 'Roll_D2', 'Roll_D3'],
            bubble_values=list('0123456789'),
            bubbles_gap=32,
            labels_gap=75,
            direction='vertical',
            bubble_width=32,
            bubble_height=32
        )

        # Class
        template.field_blocks['Class'] = FieldBlock(
            name='Class',
            field_type='QTYPE_INT',
            origin=(1830, 205),
            field_labels=['Class'],
            bubble_values=list('0123456789'),
            bubbles_gap=32,
            labels_gap=50,
            direction='vertical',
            bubble_width=32,
            bubble_height=32
        )

        # Section
        template.field_blocks['Section'] = FieldBlock(
            name='Section',
            field_type='QTYPE_ALPHA',
            origin=(1945, 205),
            field_labels=['Section'],
            bubble_values=list('ABCDEFGH'),
            bubbles_gap=32,
            labels_gap=50,
            direction='vertical',
            bubble_width=32,
            bubble_height=32
        )

        # School Code (4 digits)
        template.field_blocks['SchoolCode'] = FieldBlock(
            name='SchoolCode',
            field_type='QTYPE_INT',
            origin=(2070, 205),
            field_labels=['School_D1', 'School_D2', 'School_D3', 'School_D4'],
            bubble_values=list('0123456789'),
            bubbles_gap=32,
            labels_gap=55,
            direction='vertical',
            bubble_width=32,
            bubble_height=32
        )

        # Answer sections (Questions 1-60)
        answer_origins = [
            (120, 1650),   # Q1-10
            (350, 1650),   # Q11-20
            (580, 1650),   # Q21-30
            (810, 1650),   # Q31-40
            (1040, 1650),  # Q41-50
            (1270, 1650),  # Q51-60
        ]

        for col_idx, origin in enumerate(answer_origins):
            start_q = col_idx * 10 + 1
            end_q = start_q + 10

            template.field_blocks[f'Answers_{start_q}_{end_q - 1}'] = FieldBlock(
                name=f'Answers_{start_q}_{end_q - 1}',
                field_type='QTYPE_MCQ4',
                origin=origin,
                field_labels=[f'q{i}' for i in range(start_q, end_q)],
                bubble_values=['A', 'B', 'C', 'D'],
                bubbles_gap=48,
                labels_gap=72,
                direction='horizontal',
                bubble_width=32,
                bubble_height=32
            )

        return template

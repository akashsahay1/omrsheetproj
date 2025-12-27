"""
Excel export service
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from src.models.result import ProcessingResult, EvaluationResult, BatchResult
from src.models.candidate import Candidate
from src.utils.logging_utils import get_logger

logger = get_logger('excel_exporter')


def get_academic_year() -> str:
    """Get current academic year in format YYYY-YY (e.g., 2025-26)"""
    now = datetime.now()
    year = now.year
    month = now.month

    # Academic year starts in April
    if month >= 4:  # April onwards - current year to next year
        start_year = year
    else:  # Jan-Mar - previous year to current year
        start_year = year - 1

    end_year_short = str(start_year + 1)[-2:]  # Last 2 digits of end year
    return f"{start_year}-{end_year_short}"


class ExcelExporter:
    """
    Export processing results to Excel format
    """

    def __init__(self):
        # Styles
        self.header_font = Font(bold=True, size=11)
        self.header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        self.header_font_white = Font(bold=True, size=11, color="FFFFFF")
        self.center_align = Alignment(horizontal="center", vertical="center")
        self.left_align = Alignment(horizontal="left", vertical="center")
        self.thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        # Conditional formatting colors
        self.green_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
        self.red_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
        self.yellow_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")

        # Confidence level colors
        self.conf_high_fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # Green
        self.conf_high_font = Font(color="006100", bold=True)  # Dark green text
        self.conf_medium_fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")  # Yellow
        self.conf_medium_font = Font(color="9C5700", bold=True)  # Dark yellow/orange text
        self.conf_low_fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")  # Red
        self.conf_low_font = Font(color="9C0006", bold=True)  # Dark red text

    def export(self,
               evaluation_results: List[EvaluationResult],
               output_path: str,
               include_summary: bool = True,
               include_answers: bool = True,
               exam_level: str = 'Prarambhik',
               exam_type: str = 'Chayan') -> bool:
        """
        Export results to Excel file

        Args:
            evaluation_results: List of evaluation results
            output_path: Output file path
            include_summary: Include summary statistics
            include_answers: Include individual question answers
            exam_level: Exam level (Anshik/Mukhya)
            exam_type: Exam type (Chayan/Khoj)

        Returns:
            True if export successful
        """
        try:
            wb = Workbook()

            # Main results sheet
            self._create_results_sheet(wb, evaluation_results, include_answers,
                                      exam_level, exam_type)

            # Summary sheet (optional)
            if include_summary and evaluation_results:
                self._create_summary_sheet(wb, evaluation_results)

            # Save workbook
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            wb.save(output_path)

            logger.info(f"Exported {len(evaluation_results)} results to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def _create_results_sheet(self,
                               wb: Workbook,
                               results: List[EvaluationResult],
                               include_answers: bool,
                               exam_level: str = 'Prarambhik',
                               exam_type: str = 'Chayan') -> None:
        """Create main results sheet with organization header"""
        ws = wb.active
        ws.title = "Results"

        # Merge headers only across main data columns (A to O = 15 columns)
        # This keeps headers centered and visible without scrolling
        header_merge_col = 'O'  # S.No. to Rank (15 columns including Filename and Confidence)

        # ========== ORGANIZATION HEADER ROWS ==========

        # Row 1: Organization Name (Bold, Large, Centered)
        ws.merge_cells(f'A1:{header_merge_col}1')
        cell_org = ws.cell(row=1, column=1, value="VIDYA VIKAS SAMITI, JHARKHAND")
        cell_org.font = Font(bold=True, size=16)
        cell_org.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[1].height = 25

        # Row 2: Address (Centered)
        ws.merge_cells(f'A2:{header_merge_col}2')
        cell_addr = ws.cell(row=2, column=1, value="Shukla Colony, Hinoo, Ranchi")
        cell_addr.font = Font(size=12)
        cell_addr.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[2].height = 20

        # Row 3: Empty

        # Row 4: Exam Title (Bold, Centered) - English only
        academic_year = get_academic_year()
        exam_title = f"Vidya Bharti Pratibha {exam_type} Pratiyogita {exam_level} Pariksha {academic_year}"

        ws.merge_cells(f'A4:{header_merge_col}4')
        cell_title = ws.cell(row=4, column=1, value=exam_title)
        cell_title.font = Font(bold=True, size=14)
        cell_title.alignment = Alignment(horizontal="center", vertical="center")
        ws.row_dimensions[4].height = 25

        # Row 5: Empty
        # Row 6: Empty

        # ========== COLUMN HEADERS (Row 7) ==========
        header_row = 7

        # Column order with Confidence after Filename:
        # S.No., Filename, Confidence, Name, School Code, Class, Section, Roll No., Total Score, Percentage,
        # Correct, Wrong, Unanswered, Multi-Marked, Rank, Q1-Q60
        headers = [
            "S.No.", "Filename", "Conf %", "Name", "School Code", "Class", "Section", "Roll No.",
            "Total Score", "Percentage", "Correct", "Wrong", "Unanswered",
            "Multi-Marked", "Rank"
        ]

        # Add question columns if requested
        if include_answers:
            for q in range(1, 61):
                headers.append(f"Q{q}")

        # Write headers
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=header_row, column=col, value=header)
            cell.font = self.header_font_white
            cell.fill = self.header_fill
            cell.alignment = self.center_align
            cell.border = self.thin_border

        # ========== DATA ROWS (Starting Row 8) ==========
        data_start_row = 8

        for row_idx, result in enumerate(results, data_start_row):
            candidate = result.candidate

            if candidate is None:
                continue

            data_row = row_idx
            serial_no = row_idx - data_start_row + 1

            # Extract filename from source_image
            import os
            filename = os.path.basename(candidate.source_image) if candidate.source_image else ''

            # Get detection confidence
            confidence = candidate.detection_confidence if hasattr(candidate, 'detection_confidence') else 100.0

            # Column order with Confidence after Filename:
            # 1: S.No.
            ws.cell(row=data_row, column=1, value=serial_no)
            # 2: Filename
            ws.cell(row=data_row, column=2, value=filename)
            # 3: Confidence % (with color coding)
            conf_cell = ws.cell(row=data_row, column=3, value=f"{confidence:.0f}%")
            conf_cell.alignment = self.center_align
            # Apply color based on confidence level
            if confidence >= 80:
                conf_cell.fill = self.conf_high_fill
                conf_cell.font = self.conf_high_font
            elif confidence >= 50:
                conf_cell.fill = self.conf_medium_fill
                conf_cell.font = self.conf_medium_font
            else:
                conf_cell.fill = self.conf_low_fill
                conf_cell.font = self.conf_low_font
            # 4: Name
            ws.cell(row=data_row, column=4, value=candidate.name)
            # 5: School Code
            ws.cell(row=data_row, column=5, value=candidate.school_code)
            # 6: Class
            ws.cell(row=data_row, column=6, value=candidate.class_)
            # 7: Section
            ws.cell(row=data_row, column=7, value=candidate.section)
            # 8: Roll No.
            ws.cell(row=data_row, column=8, value=candidate.roll_number)
            # 9: Total Score
            ws.cell(row=data_row, column=9, value=result.total_score)
            # 10: Percentage
            ws.cell(row=data_row, column=10, value=f"{result.percentage}%")
            # 11: Correct
            ws.cell(row=data_row, column=11, value=result.correct_count)
            # 12: Wrong
            ws.cell(row=data_row, column=12, value=result.wrong_count)
            # 13: Unanswered
            ws.cell(row=data_row, column=13, value=result.unanswered_count)
            # 14: Multi-Marked
            ws.cell(row=data_row, column=14, value=result.multi_marked_count)
            # 15: Rank
            ws.cell(row=data_row, column=15, value=result.rank)

            # Apply formatting to all data columns
            for col in range(1, 16):
                cell = ws.cell(row=data_row, column=col)
                cell.border = self.thin_border
                if col not in (2, 4):  # All centered except Filename and Name columns
                    cell.alignment = self.center_align

            # Add answers if requested (columns 16-75 for Q1-Q60)
            if include_answers:
                for q in range(1, 61):
                    col = 15 + q
                    answer = candidate.get_answer(q)
                    cell = ws.cell(row=data_row, column=col, value=answer)
                    cell.alignment = self.center_align
                    cell.border = self.thin_border

                    # Color code multi-marked
                    if answer == "MULTI":
                        cell.fill = self.yellow_fill

        # Auto-adjust column widths
        self._auto_adjust_columns(ws)

        # Freeze header row (row 7)
        ws.freeze_panes = f"A{data_start_row}"

    def _create_summary_sheet(self,
                               wb: Workbook,
                               results: List[EvaluationResult]) -> None:
        """Create summary statistics sheet"""
        ws = wb.create_sheet("Summary")

        # Calculate statistics
        scores = [r.total_score for r in results]
        correct_counts = [r.correct_count for r in results]

        stats = [
            ("Total Candidates", len(results)),
            ("", ""),
            ("Score Statistics", ""),
            ("Highest Score", max(scores) if scores else 0),
            ("Lowest Score", min(scores) if scores else 0),
            ("Average Score", round(sum(scores) / len(scores), 2) if scores else 0),
            ("", ""),
            ("Performance Summary", ""),
            ("Avg Correct Answers", round(sum(correct_counts) / len(correct_counts), 1) if correct_counts else 0),
            ("", ""),
            ("Score Distribution", ""),
            ("50-60 marks", sum(1 for s in scores if 50 <= s <= 60)),
            ("40-50 marks", sum(1 for s in scores if 40 <= s < 50)),
            ("30-40 marks", sum(1 for s in scores if 30 <= s < 40)),
            ("20-30 marks", sum(1 for s in scores if 20 <= s < 30)),
            ("10-20 marks", sum(1 for s in scores if 10 <= s < 20)),
            ("0-10 marks", sum(1 for s in scores if 0 <= s < 10)),
            ("Negative marks", sum(1 for s in scores if s < 0)),
            ("", ""),
            ("Export Details", ""),
            ("Generated on", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]

        for row_idx, (label, value) in enumerate(stats, 1):
            cell_label = ws.cell(row=row_idx, column=1, value=label)
            cell_value = ws.cell(row=row_idx, column=2, value=value)

            if label and not value:
                # Section header
                cell_label.font = Font(bold=True, size=12)
            elif label:
                cell_label.font = Font(bold=True)

        # Adjust widths
        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 20

    def _auto_adjust_columns(self, ws) -> None:
        """Auto-adjust column widths based on content"""
        for column_cells in ws.columns:
            max_length = 0
            column = None

            # Find the column letter from a non-merged cell
            for cell in column_cells:
                if hasattr(cell, 'column_letter'):
                    column = cell.column_letter
                    break
                elif hasattr(cell, 'column'):
                    column = get_column_letter(cell.column)
                    break

            if column is None:
                continue

            for cell in column_cells:
                try:
                    if hasattr(cell, 'value') and cell.value:
                        cell_length = len(str(cell.value))
                        if cell_length > max_length:
                            max_length = cell_length
                except:
                    pass

            # Set width with padding
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column].width = max(adjusted_width, 8)

    def export_simple(self,
                      candidates: List[Candidate],
                      output_path: str) -> bool:
        """
        Export candidates without scoring (extraction only)

        Args:
            candidates: List of extracted candidates
            output_path: Output file path

        Returns:
            True if export successful
        """
        try:
            wb = Workbook()
            ws = wb.active
            ws.title = "Extracted Data"

            # Headers with Confidence after Filename
            headers = [
                "S.No.", "Filename", "Conf %", "Name", "Roll No.", "Class", "Section", "School Code"
            ]
            for q in range(1, 61):
                headers.append(f"Q{q}")

            # Write headers
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=1, column=col, value=header)
                cell.font = self.header_font_white
                cell.fill = self.header_fill
                cell.alignment = self.center_align

            # Write data
            for row_idx, candidate in enumerate(candidates, 2):
                import os
                filename = os.path.basename(candidate.source_image) if candidate.source_image else ''
                confidence = candidate.detection_confidence if hasattr(candidate, 'detection_confidence') else 100.0

                ws.cell(row=row_idx, column=1, value=row_idx - 1)
                ws.cell(row=row_idx, column=2, value=filename)
                # 3: Confidence % with color coding
                conf_cell = ws.cell(row=row_idx, column=3, value=f"{confidence:.0f}%")
                conf_cell.alignment = self.center_align
                if confidence >= 80:
                    conf_cell.fill = self.conf_high_fill
                    conf_cell.font = self.conf_high_font
                elif confidence >= 50:
                    conf_cell.fill = self.conf_medium_fill
                    conf_cell.font = self.conf_medium_font
                else:
                    conf_cell.fill = self.conf_low_fill
                    conf_cell.font = self.conf_low_font
                ws.cell(row=row_idx, column=4, value=candidate.name)
                ws.cell(row=row_idx, column=5, value=candidate.roll_number)
                ws.cell(row=row_idx, column=6, value=candidate.class_)
                ws.cell(row=row_idx, column=7, value=candidate.section)
                ws.cell(row=row_idx, column=8, value=candidate.school_code)

                for q in range(1, 61):
                    ws.cell(row=row_idx, column=8 + q, value=candidate.get_answer(q))

            self._auto_adjust_columns(ws)
            ws.freeze_panes = "A2"

            wb.save(output_path)
            logger.info(f"Exported {len(candidates)} candidates to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Simple export failed: {e}")
            return False

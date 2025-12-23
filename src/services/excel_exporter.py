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

    def export(self,
               evaluation_results: List[EvaluationResult],
               output_path: str,
               include_summary: bool = True,
               include_answers: bool = True) -> bool:
        """
        Export results to Excel file

        Args:
            evaluation_results: List of evaluation results
            output_path: Output file path
            include_summary: Include summary statistics
            include_answers: Include individual question answers

        Returns:
            True if export successful
        """
        try:
            wb = Workbook()

            # Main results sheet
            self._create_results_sheet(wb, evaluation_results, include_answers)

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
                               include_answers: bool) -> None:
        """Create main results sheet"""
        ws = wb.active
        ws.title = "Results"

        # Headers
        headers = [
            "S.No.", "Name", "Roll No.", "Class", "Section", "School Code",
            "Correct", "Wrong", "Unanswered", "Multi-Marked",
            "Total Score", "Percentage", "Rank"
        ]

        # Add question columns if requested
        if include_answers:
            for q in range(1, 61):
                headers.append(f"Q{q}")

        # Write headers
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = self.header_font_white
            cell.fill = self.header_fill
            cell.alignment = self.center_align
            cell.border = self.thin_border

        # Write data
        for row_idx, result in enumerate(results, 2):
            candidate = result.candidate

            if candidate is None:
                continue

            # Basic info
            ws.cell(row=row_idx, column=1, value=row_idx - 1)  # S.No.
            ws.cell(row=row_idx, column=2, value=candidate.name)
            ws.cell(row=row_idx, column=3, value=candidate.roll_number)
            ws.cell(row=row_idx, column=4, value=candidate.class_)
            ws.cell(row=row_idx, column=5, value=candidate.section)
            ws.cell(row=row_idx, column=6, value=candidate.school_code)

            # Scores
            ws.cell(row=row_idx, column=7, value=result.correct_count)
            ws.cell(row=row_idx, column=8, value=result.wrong_count)
            ws.cell(row=row_idx, column=9, value=result.unanswered_count)
            ws.cell(row=row_idx, column=10, value=result.multi_marked_count)
            ws.cell(row=row_idx, column=11, value=result.total_score)
            ws.cell(row=row_idx, column=12, value=f"{result.percentage}%")
            ws.cell(row=row_idx, column=13, value=result.rank)

            # Apply formatting
            for col in range(1, 14):
                cell = ws.cell(row=row_idx, column=col)
                cell.border = self.thin_border
                if col in [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
                    cell.alignment = self.center_align

            # Add answers if requested
            if include_answers:
                for q in range(1, 61):
                    col = 13 + q
                    answer = candidate.get_answer(q)
                    cell = ws.cell(row=row_idx, column=col, value=answer)
                    cell.alignment = self.center_align
                    cell.border = self.thin_border

                    # Color code
                    if answer == "MULTI":
                        cell.fill = self.yellow_fill

        # Auto-adjust column widths
        self._auto_adjust_columns(ws)

        # Freeze header row
        ws.freeze_panes = "A2"

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
            column = column_cells[0].column_letter

            for cell in column_cells:
                try:
                    if cell.value:
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

            # Headers
            headers = [
                "S.No.", "Name", "Roll No.", "Class", "Section", "School Code"
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
                ws.cell(row=row_idx, column=1, value=row_idx - 1)
                ws.cell(row=row_idx, column=2, value=candidate.name)
                ws.cell(row=row_idx, column=3, value=candidate.roll_number)
                ws.cell(row=row_idx, column=4, value=candidate.class_)
                ws.cell(row=row_idx, column=5, value=candidate.section)
                ws.cell(row=row_idx, column=6, value=candidate.school_code)

                for q in range(1, 61):
                    ws.cell(row=row_idx, column=6 + q, value=candidate.get_answer(q))

            self._auto_adjust_columns(ws)
            ws.freeze_panes = "A2"

            wb.save(output_path)
            logger.info(f"Exported {len(candidates)} candidates to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Simple export failed: {e}")
            return False

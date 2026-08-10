"""
Shared Excel output styling, used by every algorithm/runner file
(gmm_bandit, gmm_submodular, two_stage, and run_proposed_methods.py).

Previously _style_sheet was defined independently in
gmm_2class_bandit_asymmetric.py AND gmm_2class_submodular_asymmetric.py
(identical bodies, just duplicated), and two_stage_runner.py cross-imported
the bandit copy directly (two_stage_asymmetric.py itself never defined its
own). Consolidated here instead, since it's a generic openpyxl formatting
helper with no method-specific logic -- nothing about it is bandit-,
submodular-, or two_stage-specific.
"""

from __future__ import annotations

from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter


def _style_sheet(ws):
    HEADER_FILL = PatternFill("solid", fgColor="2F5496")
    HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
    ALT_FILL = PatternFill("solid", fgColor="DCE6F1")
    BORDER_SIDE = Side(style="thin", color="B8CCE4")
    CELL_BORDER = Border(left=BORDER_SIDE, right=BORDER_SIDE,
                          top=BORDER_SIDE, bottom=BORDER_SIDE)

    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = CELL_BORDER

    for row_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):
        fill = ALT_FILL if row_idx % 2 == 0 else PatternFill()
        for cell in row:
            cell.fill = fill
            cell.border = CELL_BORDER
            cell.alignment = Alignment(horizontal="center", vertical="center")
            if isinstance(cell.value, float):
                cell.number_format = "0.0000"

    for col in ws.columns:
        max_len = max(
            len(str(cell.value)) if cell.value is not None else 0 for cell in col
        )
        ws.column_dimensions[get_column_letter(col[0].column)].width = max(max_len + 4, 14)
    ws.freeze_panes = "A2"

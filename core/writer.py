"""Write a pandas DataFrame to a clean Excel file."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

if TYPE_CHECKING:
    from core.recipe_engine import Recipe

from core.target_schema import TABLE_SPEC

logger = logging.getLogger(__name__)


def _recipe_to_dataframe(recipe: Recipe) -> pd.DataFrame:
    """Convert a Recipe into a flat DataFrame for the metadata sheet."""
    rows = []
    for t in recipe.transformations:
        rule_str = json.dumps(t.rule) if t.rule else ""
        alt_str = ", ".join(t.alternatives) if t.alternatives else ""
        rows.append({
            "Target Column": t.target,
            "Type": t.type.value,
            "Source": (
                ", ".join(t.source) if isinstance(t.source, list) else (t.source or "")
            ),
            "Rule": rule_str,
            "Target Type": t.target_type or "",
            "Confidence": t.confidence if t.confidence is not None else "",
            "Alternatives": alt_str,
        })
    return pd.DataFrame(rows)


def _apply_excel_formatting(output_path: Path) -> None:
    """Apply Excel formatting to columns based on target schema."""
    if not output_path.exists():
        logger.warning(f"Output file does not exist: {output_path}")
        return
    
    try:
        # Load the workbook and get the data sheet
        wb = load_workbook(str(output_path))
        if "Data" not in wb.sheetnames:
            logger.warning(f"Data sheet not found in workbook: {wb.sheetnames}")
            return
            
        ws = wb["Data"]
        
        # Get header row to find column indices
        header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
        col_name_to_idx = {col_name: idx + 1 for idx, col_name in enumerate(header_row) if col_name}
        logger.info(f"Found columns in Excel: {list(col_name_to_idx.keys())}")
        
        # Apply formatting to each column
        formats_applied = 0
        for col in TABLE_SPEC.columns:
            if col.excel_format and col.name in col_name_to_idx:
                col_idx = col_name_to_idx[col.name]
                logger.info(f"Applying format '{col.excel_format}' to column '{col.name}' (index {col_idx})")
                
                # Apply format to entire column (skip header row)
                cells_formatted = 0
                for cell in ws[col_idx]:
                    if cell.row > 1 and cell.value is not None:  # Skip header
                        cell.number_format = col.excel_format
                        cells_formatted += 1
                
                formats_applied += 1
                logger.info(f"Formatted {cells_formatted} cells in column '{col.name}'")
            elif col.excel_format:
                logger.warning(f"Column '{col.name}' has format '{col.excel_format}' but not found in Excel")
        
        # Save the workbook
        wb.save(str(output_path))
        logger.info(f"Applied {formats_applied} Excel formats to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to apply Excel formatting: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def _apply_excel_formatting_fixed(output_path: Path) -> None:
    """Apply Excel formatting to columns based on target schema - FIXED VERSION."""
    if not output_path.exists():
        logger.warning(f"Output file does not exist: {output_path}")
        return

    try:
        # Load the workbook and get the data sheet
        wb = load_workbook(str(output_path))
        if "Data" not in wb.sheetnames:
            logger.warning(f"Data sheet not found in workbook: {wb.sheetnames}")
            return

        ws = wb["Data"]

        # Get header row to find column indices
        header_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
        col_name_to_idx = {col_name: idx + 1 for idx, col_name in enumerate(header_row) if col_name}
        logger.info(f"Found columns in Excel: {list(col_name_to_idx.keys())}")

        # Apply formatting to each column - FIXED APPROACH
        formats_applied = 0
        for col in TABLE_SPEC.columns:
            if col.excel_format and col.name in col_name_to_idx:
                col_idx = col_name_to_idx[col.name]
                column_letter = get_column_letter(col_idx)
                logger.info(
                    f"Applying format '{col.excel_format}' to column '{col.name}' (index {col_idx}, letter {column_letter})")

                # Apply format using column letter - MORE RELIABLE
                cells_formatted = 0
                for row in range(2, ws.max_row + 1):  # Skip header row (row 1)
                    cell = ws[f"{column_letter}{row}"]
                    if cell.value is not None:
                        original_format = cell.number_format
                        cell.number_format = col.excel_format
                        cells_formatted += 1

                        # Debug: Log first few cells being formatted
                        if cells_formatted <= 3:
                            logger.info(
                                f"  Cell {cell.coordinate} ({cell.value}): {original_format} -> {col.excel_format}")

                formats_applied += 1
                logger.info(f"Formatted {cells_formatted} cells in column '{col.name}'")
            elif col.excel_format:
                logger.warning(f"Column '{col.name}' has format '{col.excel_format}' but not found in Excel")

        # Save the workbook
        wb.save(str(output_path))
        logger.info(f"Applied {formats_applied} Excel formats to {output_path}")

    except Exception as e:
        logger.error(f"Failed to apply Excel formatting: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def write_excel(
    df: pd.DataFrame,
    output_path: str | Path,
    sheet_name: str = "Data",
    recipe: Recipe | None = None,
    apply_formatting: bool = True,
) -> Path:
    """Write *df* to an .xlsx file.

    If *recipe* is provided, a second sheet named "Transformations" is added
    with the column mapping details.

    Creates parent directories if they don't exist.  Returns the resolved
    output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Debug: Check DataFrame columns
    logger.info(f"DataFrame columns: {list(df.columns)}")
    logger.info(f"DataFrame dtypes: {dict(df.dtypes)}")
    logger.info(f"DataFrame shape: {df.shape}")
    
    # Ensure DataFrame has a clean index (no named index)
    df_to_write = df.reset_index(drop=True)
    logger.info(f"Writing DataFrame with shape {df_to_write.shape} to {output_path}")

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        df_to_write.to_excel(writer, sheet_name=sheet_name, index=False)

        if recipe is not None:
            recipe_df = _recipe_to_dataframe(recipe)
            recipe_df.to_excel(
                writer, sheet_name="Transformations", index=False
            )

    # Apply Excel formatting only if requested
    if apply_formatting:
        # _apply_excel_formatting(output_path)
        _apply_excel_formatting_fixed(output_path)

    return output_path
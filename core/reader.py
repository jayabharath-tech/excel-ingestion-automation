"""Read an Excel sheet into a raw pandas DataFrame with original types preserved.

Uses ``pd.read_excel(header=None)`` so that every row — including the real
header — is treated as data.  Column indices are integers (0, 1, 2, …).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from core.target_schema import TableSpec


@dataclass
class RawSheet:
    """Container for the raw DataFrame plus file/sheet metadata."""
    df: pd.DataFrame
    sheet_name: str
    file_path: str

    @property
    def total_rows(self) -> int:
        return len(self.df)

    @property
    def total_cols(self) -> int:
        return len(self.df.columns)

    @property
    def is_empty(self) -> bool:
        return self.total_rows == 0


def read_excel(
        filepath: str | Path,
        sheet_name: str | None = None,
        header: Optional[int] = None
) -> RawSheet:
    """Load every cell from a sheet into a DataFrame with no header inference.

    Types are preserved as-is from openpyxl (str, int, float, datetime, bool,
    NaN for empty cells).
    """
    filepath = Path(filepath)
    xl = pd.ExcelFile(filepath, engine="openpyxl")
    actual_sheet = sheet_name if sheet_name else xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=actual_sheet, header=header)
    xl.close()
    return RawSheet(df=df, sheet_name=actual_sheet, file_path=str(filepath))


def read_excel_with_schema(
    filepath: str | Path,
    sheet_name: str | None = None,
    schema: Optional['TableSpec'] = None,
) -> RawSheet:
    """Load Excel file with schema validation and type enforcement.
    
    If schema is provided, applies data_type constraints and validation.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    filepath = Path(filepath)
    xl = pd.ExcelFile(filepath, engine="openpyxl")
    actual_sheet = sheet_name if sheet_name else xl.sheet_names[0]
    
    # Read with header to get column names
    df = pd.read_excel(xl, sheet_name=actual_sheet, header=0, dtype="object")
    xl.close()
    
    # Apply schema validation if provided
    if schema:
        logger.info(f"Applying schema validation to {len(df.columns)} columns")
        
        # Apply data_type constraints
        for col in schema.columns:
            if col.name in df.columns:
                target_type = col.data_type
                logger.info(f"  Column '{col.name}' target type is {target_type} type")
                if target_type == "object":
                    # Force to string/object type
                    df[col.name] = df[col.name].astype(str)
                    logger.info(f"  Column '{col.name}' forced to object type")
                    
                elif target_type == "int64":
                    # Force to integer type
                    df[col.name] = pd.to_numeric(df[col.name], errors='coerce').astype('Int64')
                    logger.info(f"  Column '{col.name}' forced to int64 type")
                    
                elif target_type == "float64":
                    # Force to float type
                    df[col.name] = pd.to_numeric(df[col.name], errors='coerce')
                    logger.info(f"  Column '{col.name}' forced to float64 type")
                    
                elif target_type == "datetime64[ns]":
                    # Force to datetime type
                    if df[col.name].dtype == 'object':
                        df[col.name] = pd.to_datetime(df[col.name], errors='coerce')
                    logger.info(f"  Column '{col.name}' forced to datetime64[ns] type")

                elif target_type == "str":
                    df[col.name] = df[col.name].astype(str)
                    logger.info(f"  Column '{col.name}' forced to str type")

    # print(df["Dob"].head(3))
    return RawSheet(df=df, sheet_name=actual_sheet, file_path=str(filepath))

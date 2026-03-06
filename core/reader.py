"""Read an Excel sheet into a raw pandas DataFrame with original types preserved.

Uses ``pd.read_excel(header=None)`` so that every row — including the real
header — is treated as data.  Column indices are integers (0, 1, 2, …).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


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
) -> RawSheet:
    """Load every cell from a sheet into a DataFrame with no header inference.

    Types are preserved as-is from openpyxl (str, int, float, datetime, bool,
    NaN for empty cells).
    """
    filepath = Path(filepath)
    xl = pd.ExcelFile(filepath, engine="openpyxl")
    actual_sheet = sheet_name if sheet_name else xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=actual_sheet, header=None)
    xl.close()
    return RawSheet(df=df, sheet_name=actual_sheet, file_path=str(filepath))
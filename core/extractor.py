"""Orchestrator that chains all detection stages into a single extract call.

    read_excel  →  detect_header  →  build_profile  →  classify_rows
                →  find_boundaries  →  (optional LLM fallback)  →  DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import CONFIDENCE_THRESHOLD, LLM_ENABLED
from core.boundary_finder import DataBoundary, find_boundaries
from core.data_profiler import DataProfile, build_profile
from core.header_detector import detect_header
from core.reader import RawSheet, read_excel
from core.row_classifier import classify_rows


@dataclass
class ExtractionResult:
    """Everything produced by a single extraction run."""
    sheet: RawSheet
    header_idx: int
    boundary: DataBoundary
    profile: DataProfile
    scores: pd.DataFrame
    dataframe: pd.DataFrame
    used_llm_fallback: bool = False

    def summary(self) -> str:
        return (
            f"File: {self.sheet.file_path}\n"
            f"Sheet: {self.sheet.sheet_name}\n"
            f"{self.boundary.summary()}\n"
            f"Columns: {list(self.dataframe.columns)}\n"
            f"Extracted rows: {len(self.dataframe)}\n"
            f"LLM fallback used: {self.used_llm_fallback}"
        )


def _build_dataframe(
    sheet: RawSheet,
    header_idx: int,
    boundary: DataBoundary,
) -> pd.DataFrame:
    """Slice the raw DataFrame into a clean result using detected boundaries."""
    raw = sheet.df
    header_row = raw.iloc[header_idx]
    columns = [
        str(c).strip() if pd.notna(c) else f"_col{i}"
        for i, c in enumerate(header_row)
    ]

    data = raw.iloc[boundary.first_data_row : boundary.last_data_row + 1].copy()
    data.columns = columns
    data = data.reset_index(drop=True)

    # Drop columns that are entirely empty (unnamed filler columns).
    empty_cols = [c for c in data.columns if c.startswith("_col") and data[c].isna().all()]
    if empty_cols:
        data = data.drop(columns=empty_cols)

    # Re-infer dtypes based on actual data rows only
    data = data.infer_objects()

    return data


def extract(
    filepath: str | Path,
    sheet_name: str | None = None,
    target_columns: list[str] | None = None,
) -> ExtractionResult:
    """Run the full extraction pipeline on a single Excel file/sheet."""
    sheet = read_excel(filepath, sheet_name)

    if sheet.is_empty:
        raise ValueError(f"Sheet '{sheet.sheet_name}' in {filepath} is empty.")
    # print(sheet.df)
    header_idx = detect_header(sheet.df)
    used_llm = False

    if header_idx is None:
        if LLM_ENABLED:
            from core.llm_fallback import llm_detect

            fb = llm_detect(sheet, target_columns=target_columns)
            header_idx = fb.header_row
            profile = build_profile(sheet.df, header_idx)
            scores = classify_rows(sheet.df, header_idx, profile)
            boundary = DataBoundary(fb.header_row, fb.first_data_row, fb.last_data_row, 0.6)
            used_llm = True
        else:
            raise ValueError(
                f"Could not detect a header row in {filepath}. "
                "Enable LLM fallback (LLM_ENABLED=true) or check the file."
            )
    else:
        profile = build_profile(sheet.df, header_idx)
        scores = classify_rows(sheet.df, header_idx, profile)
        boundary = find_boundaries(scores, header_idx)

        if boundary.confidence < CONFIDENCE_THRESHOLD and LLM_ENABLED:
            from core.llm_fallback import llm_detect

            fb = llm_detect(sheet, target_columns=target_columns)
            header_idx = fb.header_row
            boundary = DataBoundary(fb.header_row, fb.first_data_row, fb.last_data_row, 0.6)
            used_llm = True

    df = _build_dataframe(sheet, header_idx, boundary)

    return ExtractionResult(
        sheet=sheet,
        header_idx=header_idx,
        boundary=boundary,
        profile=profile,
        scores=scores,
        dataframe=df,
        used_llm_fallback=used_llm,
    )
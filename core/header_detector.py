"""Detect the header row in an unstructured Excel grid (pandas version).

Scans rows top-to-bottom applying four heuristic checks:
  1. Population  — at least MIN_HEADER_COLS non-empty cells.
  2. Text-dominant — >= HEADER_TEXT_RATIO of populated cells are strings.
  3. Short labels — average string length < MAX_HEADER_LABEL_LENGTH.
  4. Data follows — the next few rows have a different type signature.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from config import (
    HEADER_TEXT_RATIO,
    MAX_HEADER_LABEL_LENGTH,
    MIN_HEADER_COLS,
)


def _is_empty(val: Any) -> bool:
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def _populated(row: pd.Series) -> list[Any]:
    return [v for v in row if not _is_empty(v)]


def _is_text(val: Any) -> bool:
    return isinstance(val, str) and val.strip() != ""


def _is_numeric_or_date(val: Any) -> bool:
    if isinstance(val, bool):
        return False
    return isinstance(val, (int, float, datetime))


def _data_follows(df: pd.DataFrame, candidate_idx: int, lookahead: int = 3) -> bool:
    """Check that rows following the candidate look like data, not more headers."""
    total = len(df)
    rows_to_check = []
    for offset in range(1, lookahead + 1):
        idx = candidate_idx + offset
        if idx < total:
            rows_to_check.append(df.iloc[idx])

    if not rows_to_check:
        return False

    candidate_pop = _populated(df.iloc[candidate_idx])
    candidate_numdate = (
        sum(1 for v in candidate_pop if _is_numeric_or_date(v)) / len(candidate_pop)
        if candidate_pop else 0
    )

    non_empty_rows = [r for r in rows_to_check if _populated(r)]
    if not non_empty_rows:
        return False

    def _numdate_ratio(row: pd.Series) -> float:
        pop = _populated(row)
        return sum(1 for v in pop if _is_numeric_or_date(v)) / len(pop) if pop else 0

    avg_numdate = sum(_numdate_ratio(r) for r in non_empty_rows) / len(non_empty_rows)
    return avg_numdate > candidate_numdate or avg_numdate > 0.1


def detect_header(df: pd.DataFrame) -> int | None:
    """Return the 0-based row index of the header, or None if not found."""
    best: tuple[int, int] | None = None

    for idx in range(len(df)):
        row = df.iloc[idx]
        populated = _populated(row)
        pop_count = len(populated)

        if pop_count < MIN_HEADER_COLS:
            continue

        text_ratio = sum(1 for v in populated if _is_text(v)) / pop_count
        if text_ratio < HEADER_TEXT_RATIO:
            continue

        texts = [v for v in populated if _is_text(v)]
        avg_len = sum(len(str(t)) for t in texts) / len(texts) if texts else 0
        if avg_len > MAX_HEADER_LABEL_LENGTH:
            continue

        if not _data_follows(df, idx):
            continue

        if best is None or pop_count > best[1]:
            best = (idx, pop_count)

    return best[0] if best else None
"""Determine the first and last data rows from a scored classification DataFrame.

Forward scan — find the first row classified as data.
Gap-tolerant scan — walk forward, allowing up to GAP_TOLERANCE consecutive
non-data rows before declaring the data region over.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import GAP_TOLERANCE


@dataclass
class DataBoundary:
    """Inclusive, 0-based row indices of the detected data region."""
    header_row: int
    first_data_row: int
    last_data_row: int
    confidence: float

    @property
    def data_row_count(self) -> int:
        return max(0, self.last_data_row - self.first_data_row + 1)

    def summary(self) -> str:
        return (
            f"Header: row {self.header_row} | "
            f"Data: rows {self.first_data_row}\u2013{self.last_data_row} "
            f"({self.data_row_count} rows) | "
            f"Confidence: {self.confidence:.2f}"
        )


def find_boundaries(
    scores_df: pd.DataFrame,
    header_idx: int,
    gap_tolerance: int = GAP_TOLERANCE,
) -> DataBoundary:
    """Locate the first and last data rows from pre-computed row scores.

    *scores_df* must be indexed by the original 0-based row number (as
    produced by ``classify_rows``).
    """
    if scores_df.empty or not scores_df["is_data"].any():
        return DataBoundary(header_idx, header_idx + 1, header_idx, 0.0)

    is_data = scores_df["is_data"]

    # Forward pass — first data row
    first_data = int(is_data[is_data].index[0])

    # Gap-tolerant forward scan
    last_data = first_data
    consecutive_non_data = 0

    for idx in scores_df.index:
        if idx < first_data:
            continue
        if is_data[idx]:
            last_data = int(idx)
            consecutive_non_data = 0
        else:
            consecutive_non_data += 1
            if consecutive_non_data > gap_tolerance:
                break

    # Confidence — average score within the detected range
    range_scores = scores_df.loc[first_data:last_data, "score"]
    confidence = round(float(range_scores.mean()), 4) if len(range_scores) else 0.0

    return DataBoundary(header_idx, first_data, last_data, confidence)
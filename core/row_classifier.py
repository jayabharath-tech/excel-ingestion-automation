"""Score rows against a DataProfile using vectorised pandas operations.

Returns a DataFrame with one row per input row (same index) containing:
  score, is_data, is_blank, population_score, type_score, mask_score, keyword_hit

Keyword detection is a hard cap — if any cell contains a summary keyword the
score is forced below the data threshold.
"""

from __future__ import annotations

import pandas as pd

from config import DATA_SCORE_THRESHOLD, SUMMARY_KEYWORDS
from core.data_profiler import DataProfile, cell_type, is_empty

W_POPULATION = 0.40
W_TYPE = 0.25
W_MASK = 0.35


def _has_keyword(row: pd.Series) -> bool:
    for val in row:
        if pd.isna(val) or not isinstance(val, str):
            continue
        lowered = val.strip().lower()
        if lowered in SUMMARY_KEYWORDS:
            return True
        for kw in SUMMARY_KEYWORDS:
            if lowered.startswith(kw):
                return True
    return False


def classify_rows(
    df: pd.DataFrame,
    header_idx: int,
    profile: DataProfile,
) -> pd.DataFrame:
    """Score every row after the header and return a classification DataFrame.

    The returned DataFrame shares the same index as the corresponding slice
    of *df* (i.e. ``header_idx + 1`` onwards).
    """
    data = df.iloc[header_idx + 1 :]

    if data.empty:
        return pd.DataFrame(
            columns=[
                "score", "is_data", "is_blank",
                "population_score", "type_score", "mask_score", "keyword_hit",
            ],
        )

    # -- Non-empty mask (True where cell has a real value) -----------------
    non_empty = data.apply(lambda col: col.map(lambda v: not is_empty(v)))

    # -- Population score (vectorised, quadratic penalty) ------------------
    pop_counts = non_empty.sum(axis=1)

    if profile.typical_populated_count > 0:
        pop_ratio = (pop_counts / profile.typical_populated_count).clip(upper=1.0)
    else:
        pop_ratio = pd.Series(0.0, index=data.index)

    pop_score = pop_ratio ** 2

    # -- Type score (per-column match, weighted by coverage) ---------------
    type_matches = pd.Series(0, index=data.index, dtype=int)
    for col_idx in data.columns:
        expected = profile.dominant_types.get(int(col_idx), "empty")
        matches = data[col_idx].map(
            lambda v, et=expected: cell_type(v) == et and not is_empty(v),
        )
        type_matches = type_matches + matches.astype(int)

    expected_populated = len(profile.populated_column_mask) or profile.total_columns
    type_checked = pop_counts.replace(0, 1)
    raw_type_ratio = type_matches / type_checked
    coverage = (pop_counts / max(expected_populated, 1)).clip(upper=1.0)
    type_score = raw_type_ratio * coverage

    # -- Mask score (Jaccard similarity, per row) --------------------------
    expected_cols = profile.populated_column_mask

    def _jaccard(row_mask: pd.Series) -> float:
        row_cols = set(row_mask[row_mask].index)
        if not expected_cols and not row_cols:
            return 0.0
        if not expected_cols or not row_cols:
            return 0.0
        return len(expected_cols & row_cols) / len(expected_cols | row_cols)

    mask_score = non_empty.apply(_jaccard, axis=1)

    # -- Keyword check (row-level) ----------------------------------------
    keyword_hit = data.apply(_has_keyword, axis=1)

    # -- Combine -----------------------------------------------------------
    scores = W_POPULATION * pop_score + W_TYPE * type_score + W_MASK * mask_score
    scores[keyword_hit] = scores[keyword_hit].clip(upper=DATA_SCORE_THRESHOLD * 0.5)

    is_blank = pop_counts == 0
    is_data = scores >= DATA_SCORE_THRESHOLD

    return pd.DataFrame(
        {
            "score": scores.round(4),
            "is_data": is_data,
            "is_blank": is_blank,
            "population_score": pop_score.round(4),
            "type_score": type_score.round(4),
            "mask_score": mask_score.round(4),
            "keyword_hit": keyword_hit,
        },
    )
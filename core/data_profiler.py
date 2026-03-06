"""Build a statistical profile of the data columns using pandas.

The profile captures what a "normal" data row looks like so that later stages
can score individual rows against it.  Leverages vectorised pandas operations
for speed on large files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from config import POPULATION_RATE_THRESHOLD, PROFILE_SAMPLE_SIZE


# ── Cell-level helpers (shared with row_classifier) ──────────────────────

def is_empty(val: Any) -> bool:
    if pd.isna(val):
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    return False


def cell_type(val: Any) -> str:
    if is_empty(val):
        return "empty"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, (int, float)):
        return "numeric"
    if isinstance(val, datetime):
        return "date"
    return "text"


# ── Profile dataclass ───────────────────────────────────────────────────

@dataclass
class DataProfile:
    dominant_types: dict[int, str]
    population_rates: dict[int, float]
    typical_populated_count: float
    populated_column_mask: set[int] = field(default_factory=set)
    total_columns: int = 0
    sample_size: int = 0
    confidence: float = 1.0


# ── Public API ───────────────────────────────────────────────────────────

def build_profile(
    df: pd.DataFrame,
    header_idx: int,
    sample_size: int = PROFILE_SAMPLE_SIZE,
) -> DataProfile:
    """Profile the first *sample_size* rows after the header.

    Caps the sample at half the available rows to avoid sampling trailing
    noise (totals, summaries) that would pollute the profile.
    """
    first_data = header_idx + 1
    available = len(df) - first_data
    safe_count = max(1, available // 2)
    effective = min(sample_size, safe_count)

    sample = df.iloc[first_data : first_data + effective]
    actual_sample = len(sample)
    num_cols = len(df.columns)

    if actual_sample == 0:
        return DataProfile(
            dominant_types={},
            population_rates={},
            typical_populated_count=0,
            total_columns=num_cols,
            sample_size=0,
            confidence=0.0,
        )

    # Non-empty mask (True where a cell has a real value)
    non_empty = sample.apply(lambda col: col.map(lambda v: not is_empty(v)))

    # Populated count per row → median gives the typical count
    pop_per_row = non_empty.sum(axis=1)
    typical_pop = float(pop_per_row.median())

    # Population rate per column
    pop_rates = {int(col): float(non_empty[col].mean()) for col in sample.columns}

    # Columns populated in >= 80 % of sample rows
    mask = {col for col, rate in pop_rates.items() if rate >= POPULATION_RATE_THRESHOLD}

    # Dominant type per column (mode of non-empty cell types)
    dominant_types: dict[int, str] = {}
    for col_idx in sample.columns:
        types = sample[col_idx].map(cell_type)
        non_empty_types = types[types != "empty"]
        if non_empty_types.empty:
            dominant_types[int(col_idx)] = "empty"
        else:
            dominant_types[int(col_idx)] = str(non_empty_types.mode().iloc[0])

    # Confidence — drops when sample rows vary in shape
    spread = int(pop_per_row.max() - pop_per_row.min())
    consistency = 1.0 - (spread / max(num_cols, 1))
    confidence = max(0.0, min(1.0, consistency))

    return DataProfile(
        dominant_types=dominant_types,
        population_rates=pop_rates,
        typical_populated_count=typical_pop,
        populated_column_mask=mask,
        total_columns=num_cols,
        sample_size=actual_sample,
        confidence=confidence,
    )
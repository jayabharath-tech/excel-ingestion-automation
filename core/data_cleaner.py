"""Data cleaning entry point.

Delegates to TABLE_SPEC.clean() which runs per-column rules defined in
core/target_schema.py.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from core.target_schema import TABLE_SPEC


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Run all applicable cleaning rules. Returns (cleaned_df, applied_rule_ids)."""
    print(df["AnnualIncome"])
    print(df["MonthlyIncome"])
    return TABLE_SPEC.clean(df)
"""Write a pandas DataFrame to a clean Excel file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from core.recipe_engine import Recipe


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


def write_excel(
    df: pd.DataFrame,
    output_path: str | Path,
    sheet_name: str = "Data",
    recipe: Recipe | None = None,
) -> Path:
    """Write *df* to an .xlsx file.

    If *recipe* is provided, a second sheet named "Transformations" is added
    with the column mapping details.

    Creates parent directories if they don't exist.  Returns the resolved
    output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        if recipe is not None:
            recipe_df = _recipe_to_dataframe(recipe)
            recipe_df.to_excel(
                writer, sheet_name="Transformations", index=False
            )

    return output_path
"""Data cleaning entry point.

Delegates to TABLE_SPEC.clean() which runs per-column rules defined in
core/target_schema.py.
"""

from __future__ import annotations

from typing import List

from core.target_schema import TABLE_SPEC
from core.constant import Gender, SA_PARTICLES

import pandas as pd
import numpy as np
from datetime import date
from luhncheck import is_luhn


def validate_sa_id(series: pd.Series) -> pd.Series:
    """
    Validate SA ID format + Luhn checksum.
    """
    s = series.astype(str)
    
    # Filter out None, empty, and invalid values before applying Luhn
    valid_format_mask = (
            (s.str.len() == 13) &
            (s.str.isdigit())
    )
    
    # Only apply Luhn check to properly formatted IDs
    # Use a custom function to handle NaN values
    def safe_is_luhn(id_str):
        if pd.isna(id_str) or not isinstance(id_str, str):
            return False
        return is_luhn(id_str)
    
    luhn_mask = s.where(valid_format_mask).apply(safe_is_luhn)
    
    # Combine format validation and Luhn check
    valid_mask = valid_format_mask & luhn_mask

    return valid_mask


def extract_dob(series: pd.Series) -> pd.Series:
    """
    Extract DOB from SA ID.
    """
    s = series.astype(str)

    yy = s.str[0:2].astype(int)
    mm = s.str[2:4].astype(int)
    dd = s.str[4:6].astype(int)

    current_year = date.today().year % 100

    year = np.where(
        yy <= current_year,
        2000 + yy,
        1900 + yy
    )

    dob = pd.to_datetime(
        dict(year=year, month=mm, day=dd),
        errors="coerce"
    )

    return dob


def extract_gender(series: pd.Series) -> pd.Series:
    """
    Extract gender from SA ID.
    """
    s = series.astype(str)

    gender_digits = s.str[6:10].astype(int)

    gender = np.where(
        gender_digits > 4999,
        Gender.MALE.value,
        Gender.FEMALE.value
    )

    return pd.Series(gender, index=series.index)


def correct_sa_id_fields(
        df: pd.DataFrame,
        id_column: str,
        dob_column: str,
        gender_column: str
) -> pd.DataFrame:
    """
    Correct DOB and Gender based on SA ID only when:
    - ID is valid
    - Extracted DOB is valid
    """

    # Step 1: Validate ID
    valid_id_mask = validate_sa_id(df[id_column])

    # Step 2: Extract DOB for valid IDs
    extracted_dob = extract_dob(df.loc[valid_id_mask, id_column])

    # Step 3: Check DOB validity
    valid_dob_mask = extracted_dob.notna()

    # Combine masks
    final_mask = valid_id_mask.copy()
    final_mask.loc[valid_id_mask] = valid_dob_mask

    # Step 4: Extract gender
    extracted_gender = extract_gender(df.loc[final_mask, id_column])

    # Step 5: Update dataframe
    # Store as datetime64[ns] for proper pandas date handling
    df.loc[final_mask, dob_column] = extracted_dob[valid_dob_mask]
    df.loc[final_mask, gender_column] = extracted_gender

    return df


def normalize_sa_names(df: pd.DataFrame, firstname_col: str, lastname_col: str) -> pd.DataFrame:
    df = df.copy()

    # First names: simple title case
    df[firstname_col] = (
        df[firstname_col]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # Last names: title case first
    last = (
        df[lastname_col]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # Fix common SA particles
    for p in SA_PARTICLES:
        last = last.str.replace(fr"\b{p.title()}\b", p, regex=True)

    df[lastname_col] = last

    return df

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, List[str]]:
    """Run all applicable cleaning rules. Returns (cleaned_df, applied_rule_ids)."""
    df_base = (df
               .pipe(correct_sa_id_fields, "IdNo", "Dob", "Gender")
               .pipe(normalize_sa_names, "FirstName", "Surname")
    )
    return TABLE_SPEC.clean(df_base)

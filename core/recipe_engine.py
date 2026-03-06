"""Recipe engine for column mapping and transformation."""

import datetime
import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from config import TARGET_DATE_FORMAT
from core import ExtractionResult

logger = logging.getLogger(__name__)


class TransformationType(str, Enum):
    """Types of column transformations."""

    RENAME = "rename"
    SPLIT = "split"
    MERGE = "merge"
    EMPTY = "empty"


class TransformationItem(BaseModel):
    """Single transformation item."""

    target: str = Field(description="Target column name")
    type: TransformationType = Field(description="Type of transformation")
    source: Optional[Union[str, List[str]]] = Field(
        default=None, description="Source column(s)"
    )
    rule: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None, description="Transformation rule"
    )
    target_type: Optional[str] = Field(
        default=None,
        description="Target data type: datetime, numeric, or null to keep as-is",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence score 0.0-1.0 for this mapping",
    )
    alternatives: Optional[List[str]] = Field(
        default=None,
        description="Other source columns that could also match this target",
    )


class RecipeLLMResponse(BaseModel):
    """Pydantic model for LLM recipe generation response."""

    transformations: List[TransformationItem] = Field(
        description="List of transformations"
    )


class Recipe(BaseModel):
    """Recipe for transforming source data to target schema."""

    transformations: List[TransformationItem]

    def summary(self) -> str:
        """Get a summary of the recipe."""
        lines = ["Recipe Summary:"]
        for i, transform in enumerate(self.transformations, 1):
            if transform.type == TransformationType.RENAME:
                lines.append(f"  {i}. {transform.source} → {transform.target}")
            elif transform.type == TransformationType.SPLIT:
                lines.append(
                    f"  {i}. Split {transform.source} → {transform.target} ({transform.rule})"
                )
            elif transform.type == TransformationType.MERGE:
                lines.append(
                    f"  {i}. Merge {transform.source} → {transform.target} ({transform.rule})"
                )
            elif transform.type == TransformationType.EMPTY:
                lines.append(f"  {i}. {transform.target} = empty")
        return "\n".join(lines)


def _extract_json(text: str) -> str:
    """Extract JSON from LLM response, handling markdown fences and truncation."""
    logger.info(f"Raw LLM response length: {len(text)} chars")
    logger.info(f"Raw LLM response: {text[:1000]}...")  # Debug: Log first 1000 chars

    # Remove markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    # Find first { to last }
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or start >= end:
        logger.error(f"No valid JSON found in response. Cleaned text: {text}")
        raise ValueError("No valid JSON found in response")

    json_text = text[start : end + 1]
    
    # Check if response looks truncated
    if len(text.strip()) > 0 and not text.strip().endswith('}'):
        logger.warning("Response appears to be truncated - does not end with '}'")
    
    logger.info(f"Extracted JSON: {json_text}")  # Debug: Log extracted JSON
    return json_text


def _serialize_sample(value: Any) -> Any:
    """Convert a sample value to a JSON-safe representation."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, str) and len(value) > 50:
        return value[:50] + "..."
    return value


def build_column_metadata(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna()
    total = len(s)

    if total == 0:
        return {"inferred_type": "unknown", "sample_values": []}

    # --- Use pandas inferred dtype ---
    dtype = str(series.dtype)

    if dtype.startswith("int") or dtype.startswith("float"):
        inferred_type = "numeric"
    elif dtype.startswith("datetime"):
        inferred_type = "datetime"
    else:
        inferred_type = "string"

    # --- Cardinality ---
    unique_ratio = s.nunique() / total

    # --- Sample values (reduced to 3 for smaller metadata) ---
    samples = s.drop_duplicates().head(3).tolist()
    samples = [_serialize_sample(v) for v in samples]

    # --- Simplified stats (only for string columns) ---
    if inferred_type == "string":
        s_str = s.astype(str)
        avg_length = s_str.str.len().mean()
    else:
        avg_length = 0

    return {
        "inferred_type": inferred_type,
        "unique_ratio": round(unique_ratio, 3),
        "sample_values": samples,
        "avg_length": round(avg_length, 2),
    }


def generate_recipe(
    df: pd.DataFrame,
    structure: ExtractionResult,
    target_columns: List[str],
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> Recipe:
    """
    Hybrid deterministic + LLM-based column mapping generator.
    """
    # -----------------------------------------
    # 1. Deterministic pre-matching
    # -----------------------------------------
    source_names = df.columns

    normalized_source = {_normalize(c): c for c in source_names}
    normalized_target = {_normalize(t): t for t in target_columns}

    deterministic_transformations = []

    matched_sources = set()
    matched_targets = set()

    for norm, target_original in normalized_target.items():
        if norm in normalized_source:
            source_original = normalized_source[norm]
            deterministic_transformations.append(
                TransformationItem(
                    target=target_original,
                    type=TransformationType.RENAME,
                    source=source_original,
                    rule="",
                )
            )
            matched_sources.add(source_original)
            matched_targets.add(target_original)

    # -----------------------------------------
    # 2. Identify unresolved columns
    # -----------------------------------------
    unresolved_sources = [col for col in df.columns if col not in matched_sources]

    unresolved_targets = [t for t in target_columns if t not in matched_targets]

    # If everything resolved deterministically
    if not unresolved_targets:
        return Recipe(transformations=deterministic_transformations)

    # -----------------------------------------
    # 3. Build metadata (NO RAW DATA)
    # -----------------------------------------
    source_metadata = []
    for col_name in unresolved_sources:
        series = df[col_name]
        meta = build_column_metadata(series)
        source_metadata.append({"name": col_name, "metadata": meta})

    # -----------------------------------------
    # 4. Build context of already-resolved mappings
    # -----------------------------------------
    resolved_context = ""
    if deterministic_transformations:
        lines = ["Already resolved mappings (for context, do NOT include these):"]
        for t in deterministic_transformations:
            lines.append(f'  "{t.source}" -> "{t.target}" (rename)')
        resolved_context = "\n".join(lines)

    # -----------------------------------------
    # 5. LLM Prompt (strict, minimal, safe)
    # -----------------------------------------
    
    # Log metadata size for debugging
    metadata_json = json.dumps(source_metadata, indent=2)
    logger.info(f"Source metadata size: {len(metadata_json)} chars, {len(source_metadata)} columns")
    
    system_prompt = f"""\
You are a schema alignment engine.

{resolved_context}

Source columns with metadata and sample values:
{metadata_json}

Unresolved target columns:
{json.dumps(unresolved_targets, indent=2)}

Return ONLY JSON in this format:

{{
  "transformations": [
    {{
      "target": "column_name",
      "type": "rename|split|merge|empty",
      "source": "source_name or list",
      "rule": {{ structured_rule_object_or_null }},
      "target_type": "datetime|numeric|null",
      "confidence": 0.0 to 1.0,
      "alternatives": ["other_source_col", ...] or null
    }}
  ]
}}

Transformation rules:

1. rename (DEFAULT/PREFERRED):
   - Use for ANY direct 1-to-1 column mapping
   - Use when source column name semantically matches target
   - Use even if source contains multiple words (e.g., "FIRST NAMES" -> "FirstName")
   - rule: null

2. split (USE ONLY when explicitly needed):
   - Use ONLY when target represents a PART of a source column
   - Example: "FULL_NAME" -> ["FirstName", "Surname"]
   - rule must be:
     {{
        "operation": "split",
        "delimiter": "string",
        "pick": "first|last|index|all_except_last",
        "index": integer (only if pick=index)
     }}

3. merge:
   - Use ONLY when combining multiple source columns
   - rule must be:
     {{
        "operation": "merge",
        "delimiter": "string"
     }}

4. empty:
   - Use when no matching source column exists
   - rule: null

Every transformation may also include:
  "target_type": "datetime" | "numeric" | null

- Use "datetime" when the source column contains date/time values \
that should be parsed (e.g., "2024-01-15", "15/01/2024", "Jan 15 2024").
- Use "numeric" when the source column contains numbers stored as text.
- Use null (or omit) to keep the original type.

When target_type is "datetime", you MUST also include "source_format" in the rule \
using Python strftime codes. Infer the format from:
  1. The column header name — e.g., "Invoice Date (yyyyMMdd)" implies "%Y%m%d"
  2. The sample values — e.g., "15/01/2024" implies "%d/%m/%Y"
Example rule for a datetime column:
  "rule": {{ "source_format": "%Y%m%d" }}
Common mappings: yyyyMMdd -> %Y%m%d, yyyy-MM-dd -> %Y-%m-%d, \
dd/MM/yyyy -> %d/%m/%Y, MM/dd/yyyy -> %m/%d/%Y, dd-MM-yyyy -> %d-%m-%Y.
If the source is already a native Excel date (sample values look like timestamps), \
set "rule": null — no source_format is needed.

Every transformation MUST include:
  "confidence": a float between 0.0 and 1.0

- 1.0 = exact or near-exact name match with matching data type
- 0.7-0.9 = strong semantic match (e.g., "Inv Amt" -> "InvoiceAmount")
- 0.4-0.6 = plausible but uncertain (multiple source columns could fit)
- below 0.4 = weak guess

If confidence is below 0.7, you MUST also include:
  "alternatives": ["other_source_col1", "other_source_col2"]
listing other source columns that could also match this target.
If confidence is 0.7 or above, set "alternatives": null.

CRITICAL RULES:
- PREFER "rename" over "split" whenever possible
- "FIRST NAMES" should map to "FirstName" with "rename", NOT "split"
- Do NOT hallucinate new source columns.
- Do NOT include explanations.
- Return ONLY JSON.\
"""

    # -----------------------------------------
    # 6. LLM Call
    # -----------------------------------------
    client = OpenAI(base_url=base_url, api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate transformation recipe."},
        ],
    )

    response_text = response.choices[0].message.content
    print(response_text)
    json_text = _extract_json(response_text)
    llm_response = RecipeLLMResponse.model_validate(json.loads(json_text))

    # -----------------------------------------
    # 7. Combine deterministic + LLM output
    # -----------------------------------------
    final_transformations = deterministic_transformations + llm_response.transformations

    # -----------------------------------------
    # 8. Validation Layer
    # -----------------------------------------
    logger.info("validating recipe...")
    _validate_recipe(final_transformations, source_names, target_columns)

    return Recipe(transformations=final_transformations)


def _normalize(name: str) -> str:
    return name.lower().replace("_", "").replace(" ", "").strip()


def _validate_recipe(transformations, source_columns, target_columns):
    targets = [t.target for t in transformations]

    # All targets must be covered
    missing = set(target_columns) - set(targets)
    if missing:
        raise ValueError(f"Missing mappings for targets: {missing}")

    # No unknown sources allowed
    for t in transformations:
        src = t.source
        if isinstance(src, list):
            for s in src:
                if s not in source_columns:
                    raise ValueError(f"Invalid source column: {s}")
        elif isinstance(src, str) and src:
            if src not in source_columns:
                raise ValueError(f"Invalid source column: {src}")


def apply_recipe(
    df: pd.DataFrame, recipe: Recipe, target_columns: List[str]
) -> pd.DataFrame:
    source_df = df.copy()
    target_data = {}

    for transform in recipe.transformations:
        target_col = transform.target
        source_col = transform.source
        rule = transform.rule

        # ---------------------------------
        # EMPTY
        # ---------------------------------
        if transform.type == TransformationType.EMPTY:
            target_data[target_col] = pd.Series([None] * len(source_df))

        # ---------------------------------
        # RENAME — preserve original dtype
        # ---------------------------------
        elif transform.type == TransformationType.RENAME:
            if source_col in source_df.columns:
                target_data[target_col] = source_df[source_col].reset_index(drop=True)
            else:
                target_data[target_col] = pd.Series([None] * len(source_df))

        # ---------------------------------
        # SPLIT — string operation required
        # ---------------------------------
        elif transform.type == TransformationType.SPLIT:
            source_col = transform.source

            if source_col not in source_df.columns:
                target_data[target_col] = pd.Series([None] * len(source_df))
                continue

            series = source_df[source_col].fillna("").astype(str).str.strip()

            if isinstance(rule, dict) and rule.get("operation") == "split":
                delimiter = rule.get("delimiter", " ")
                pick = rule.get("pick")

                split_series = series.str.split(delimiter)

                if pick == "first":
                    target_data[target_col] = split_series.str[0].fillna("")

                elif pick == "last":
                    target_data[target_col] = split_series.str[-1].fillna("")

                elif pick == "all_except_last":
                    target_data[target_col] = (
                        split_series.str[:-1].str.join(delimiter).fillna("")
                    )

                elif pick == "index":
                    idx = rule.get("index", 0)
                    target_data[target_col] = split_series.str[idx].fillna("")

                else:
                    target_data[target_col] = series
            else:
                target_data[target_col] = series

        # ---------------------------------
        # MERGE — string operation required
        # ---------------------------------
        elif transform.type == TransformationType.MERGE:
            source_cols = (
                transform.source
                if isinstance(transform.source, list)
                else [transform.source]
            )

            if not isinstance(rule, dict) or rule.get("operation") != "merge":
                delimiter = ""
            else:
                delimiter = rule.get("delimiter", "")

            merged = (
                source_df[source_cols]
                .fillna("")
                .astype(str)
                .agg(delimiter.join, axis=1)
            )

            target_data[target_col] = merged

        # ---------------------------------
        # Type coercion (applies to any transformation type)
        # ---------------------------------
        if target_col in target_data and transform.target_type:
            if transform.target_type == "datetime":
                source_format = (
                    rule.get("source_format")
                    if isinstance(rule, dict)
                    else None
                )
                if source_format:
                    parsed = pd.to_datetime(
                        target_data[target_col].astype(str).str.strip(),
                        format=source_format,
                        errors="coerce",
                    )
                else:
                    parsed = pd.to_datetime(
                        target_data[target_col], errors="coerce"
                    )
                target_data[target_col] = parsed.dt.strftime(
                    TARGET_DATE_FORMAT
                ).where(parsed.notna(), None)
            elif transform.target_type == "numeric":
                target_data[target_col] = pd.to_numeric(
                    target_data[target_col], errors="coerce"
                )

    for col in target_columns:
        if col not in target_data:
            target_data[col] = pd.Series([None] * len(source_df))

    return pd.DataFrame(target_data)[target_columns]


# ── Post-apply validation ────────────────────────────────────────────────

NULL_SPIKE_THRESHOLD = 0.20
CAST_FAILURE_THRESHOLD = 0.10
LOW_CONFIDENCE_THRESHOLD = 0.7


@dataclass
class ValidationWarning:
    column: str
    check: str
    message: str

    def __str__(self) -> str:
        return f'Column "{self.column}" - {self.message}'


def validate_output(
    source_df: pd.DataFrame,
    output_df: pd.DataFrame,
    recipe: Recipe,
) -> List[ValidationWarning]:
    """Run sanity checks after recipe application. Returns warnings only."""
    warnings: List[ValidationWarning] = []

    # --- Row count ---
    if len(output_df) != len(source_df):
        warnings.append(
            ValidationWarning(
                column="*",
                check="row_count",
                message=(
                    f"Row count mismatch: source={len(source_df)}, "
                    f"output={len(output_df)}"
                ),
            )
        )

    for transform in recipe.transformations:
        target_col = transform.target
        if target_col not in output_df.columns:
            continue

        target_series = output_df[target_col]

        # --- Skip expected-empty columns ---
        if transform.type == TransformationType.EMPTY:
            continue

        # --- Low confidence mapping ---
        if (
            transform.confidence is not None
            and transform.confidence < LOW_CONFIDENCE_THRESHOLD
        ):
            alt_text = ""
            if transform.alternatives:
                alt_text = f" (alternatives: {', '.join(transform.alternatives)})"
            warnings.append(
                ValidationWarning(
                    column=target_col,
                    check="low_confidence",
                    message=(
                        f"mapping confidence {transform.confidence:.0%} "
                        f"for source \"{transform.source}\"{alt_text}"
                    ),
                )
            )

        # --- Empty output column ---
        if target_series.isna().all() or (target_series.astype(str).str.strip() == "").all():
            warnings.append(
                ValidationWarning(
                    column=target_col,
                    check="empty_column",
                    message="100% of values are null/empty after transformation",
                )
            )
            continue

        # --- Null spike ---
        source_col = transform.source
        if isinstance(source_col, str) and source_col in source_df.columns:
            source_null_rate = source_df[source_col].isna().mean()
            target_null_rate = target_series.isna().mean()
            spike = target_null_rate - source_null_rate

            if spike > NULL_SPIKE_THRESHOLD:
                warnings.append(
                    ValidationWarning(
                        column=target_col,
                        check="null_spike",
                        message=(
                            f"null rate jumped from {source_null_rate:.0%} "
                            f"(source) to {target_null_rate:.0%} (target)"
                        ),
                    )
                )

        # --- Type cast failures ---
        if transform.target_type == "datetime":
            nat_count = target_series.isna().sum()
            source_non_null = 0
            if isinstance(source_col, str) and source_col in source_df.columns:
                source_non_null = source_df[source_col].notna().sum()
            if source_non_null > 0:
                failure_rate = nat_count / source_non_null
                if failure_rate > CAST_FAILURE_THRESHOLD:
                    warnings.append(
                        ValidationWarning(
                            column=target_col,
                            check="cast_failure",
                            message=(
                                f"{failure_rate:.0%} of values failed "
                                f"datetime parsing"
                            ),
                        )
                    )

        elif transform.target_type == "numeric":
            nan_count = target_series.isna().sum()
            source_non_null = 0
            if isinstance(source_col, str) and source_col in source_df.columns:
                source_non_null = source_df[source_col].notna().sum()
            if source_non_null > 0:
                failure_rate = nan_count / source_non_null
                if failure_rate > CAST_FAILURE_THRESHOLD:
                    warnings.append(
                        ValidationWarning(
                            column=target_col,
                            check="cast_failure",
                            message=(
                                f"{failure_rate:.0%} of values failed "
                                f"numeric parsing"
                            ),
                        )
                    )

    return warnings
"""Optional LLM-based fallback for structure detection.

Only invoked when heuristic confidence is below CONFIDENCE_THRESHOLD and
LLM_ENABLED is true.  Uses a top-N / bottom-N windowing strategy to keep
prompts small.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from core.reader import RawSheet


# ── Pydantic schema for LLM response ─────────────────────────────────────

class LLMStructureResponse(BaseModel):
    header_row: int = Field(description="0-based row index of the header")
    last_data_row: int = Field(description="0-based row index of the last data row")


@dataclass
class FallbackResult:
    header_row: int
    first_data_row: int
    last_data_row: int


# ── Prompt construction ──────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an Excel structure analyser.  You will receive the TOP rows and \
BOTTOM rows of a spreadsheet.  Identify the structure and return a JSON \
object matching this schema:

{schema}

Rules:
- Row indices are 0-based.
- Rows before the header are titles / metadata / blanks — not data.
- The header row is the first row where MOST cells are short text labels \
  describing the data below.
- After the data there may be totals, averages, notes, disclaimers, blank \
  rows — these are NOT data rows.  Exclude them.
- A single blank row inside the data does NOT end the data.

{target_columns_hint}

Return ONLY valid JSON.  No markdown fences, no explanation.\
"""

_TARGET_COLUMNS_HINT = """\
You are also given the expected target column names for this file.  \
The actual column names in the spreadsheet may differ (abbreviations, \
synonyms, different casing, extra/missing words) but will be semantically \
similar.  Use these to identify the correct header row — it should be the \
row whose cells best match these target column names.

Target columns: {columns}\
"""


def _fmt_cell(value: Any) -> str:
    if pd.isna(value):
        return "null"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, datetime):
        return f'"{value.isoformat()}"'
    return str(value)


def _fmt_row(idx: int, row: pd.Series) -> str:
    cells = ", ".join(_fmt_cell(c) for c in row)
    return f"Row {idx}: [{cells}]"


def _build_prompt(sheet: RawSheet, window: int = 20) -> str:
    total = sheet.total_rows
    df = sheet.df
    lines = [f"The sheet has {total} rows.\n"]

    if total <= window * 2:
        lines.append("ALL ROWS:")
        for i in range(total):
            lines.append(_fmt_row(i, df.iloc[i]))
    else:
        lines.append(f"TOP {window} ROWS:")
        for i in range(window):
            lines.append(_fmt_row(i, df.iloc[i]))

        skipped = total - window * 2
        lines.append(f"\n... ({skipped} rows omitted) ...\n")

        lines.append(f"BOTTOM {window} ROWS:")
        start = total - window
        for i in range(start, total):
            lines.append(_fmt_row(i, df.iloc[i]))

    lines.append("\nReturn the JSON.")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")
    return json.loads(text[start:end])


# ── Public API ───────────────────────────────────────────────────────────

def llm_detect(
    sheet: RawSheet,
    *,
    target_columns: list[str] | None = None,
    base_url: str = LLM_BASE_URL,
    api_key: str = LLM_API_KEY,
    model: str = LLM_MODEL,
) -> FallbackResult:
    """Ask the LLM to detect header and data boundaries."""
    schema_json = json.dumps(LLMStructureResponse.model_json_schema(), indent=2)
    hint = (
        _TARGET_COLUMNS_HINT.format(columns=", ".join(target_columns))
        if target_columns
        else ""
    )
    system = _SYSTEM_PROMPT.format(schema=schema_json, target_columns_hint=hint)
    user_prompt = _build_prompt(sheet)

    client = OpenAI(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    raw = _extract_json(resp.choices[0].message.content)
    parsed = LLMStructureResponse.model_validate(raw)

    return FallbackResult(
        header_row=parsed.header_row,
        first_data_row=parsed.header_row + 1,
        last_data_row=parsed.last_data_row,
    )
"""Central configuration for the Excel table extractor."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Worker Process ──────────────────────────────────────────────────────────
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 2))
OUTPUT_WORKERS = int(os.getenv("OUTPUT_WORKERS", 2))

# ── Folder paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
SOURCE_DIR = Path(os.getenv("SOURCE_DIR", BASE_DIR / "data" / "source"))
PROCESS_DIR = Path(os.getenv("PROCESS_DIR", BASE_DIR / "data" / "process"))
TARGET_DIR = Path(os.getenv("TARGET_DIR", BASE_DIR / "data" / "target"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", BASE_DIR / "data" / "output"))
COMPLETED_DIR = Path(os.getenv("COMPLETED_DIR", BASE_DIR / "data" / "completed"))
ERROR_DIR = Path(os.getenv("ERROR_DIR", BASE_DIR / "data" / "error"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", BASE_DIR / "logs"))

# ── Header detection ──────────────────────────────────────────────────────
MIN_HEADER_COLS = 3
HEADER_TEXT_RATIO = 0.7
MAX_HEADER_LABEL_LENGTH = 50

# ── Data profiling ────────────────────────────────────────────────────────
PROFILE_SAMPLE_SIZE = 20
POPULATION_RATE_THRESHOLD = 0.8

# ── Row classification ───────────────────────────────────────────────────
DATA_SCORE_THRESHOLD = 0.5
SUMMARY_KEYWORDS = frozenset({
    "total", "sum", "average", "avg", "subtotal", "sub-total",
    "grand total", "count", "mean", "min", "max",
    "summary", "note", "disclaimer", "notes",
})

# ── Boundary detection ───────────────────────────────────────────────────
GAP_TOLERANCE = 1

# ── Confidence / LLM fallback ────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.7
LLM_ENABLED = os.getenv("LLM_ENABLED", "false").lower() == "true"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

#  ── Target Columns ──────────────────────────────────────────────────────────
TARGET_DATE_FORMAT = os.getenv("TARGET_DATE_FORMAT", "%d/%m/%Y")

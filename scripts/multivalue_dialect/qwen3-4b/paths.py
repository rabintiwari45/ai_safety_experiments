"""Paths and model id for HarmBench standard dialect probes on Qwen3-4B."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/multivalue_dialect/qwen3-4b/<file>.py -> scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import (  # noqa: E402
    HARMBENCH_STANDARD_DIR,
    PROJECT_ROOT,
    THINKING_SAMPLING_PARAMS,
)

MODEL_PATH = "Qwen/Qwen3-4B"
MAX_NEW_TOKENS = 12048
SAMPLING_PARAMS = THINKING_SAMPLING_PARAMS

MULTIVALUE_STANDARD_DIR = (
    PROJECT_ROOT / "outputs/multivalue_dialect/harmbench/standard"
)
# Dialect text is shared. Qwen3-8B already holds probes_dialects.jsonl.
PROBES = MULTIVALUE_STANDARD_DIR / "qwen3-8b" / "probes_dialects.jsonl"
OUTPUT_DIR = MULTIVALUE_STANDARD_DIR / "qwen3-4b"
BASELINE = HARMBENCH_STANDARD_DIR / "qwen3-4b" / "baseline.jsonl"

DEFAULT_INFER = OUTPUT_DIR / "probes_infer_african_american_vernacular.jsonl"
DEFAULT_INFER_CLASSIFIED = (
    OUTPUT_DIR / "probes_infer_african_american_vernacular_classified.jsonl"
)
SINGAPORE_INFER = OUTPUT_DIR / "probes_infer_colloquial_singapore.jsonl"
SINGAPORE_INFER_CLASSIFIED = (
    OUTPUT_DIR / "probes_infer_colloquial_singapore_classified.jsonl"
)

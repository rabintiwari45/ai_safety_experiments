"""Paths for StrongREJECT dialect probes on Qwen3-8B."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/multivalue_dialect/strongreject/qwen3/<file>.py -> scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import PROJECT_ROOT, THINKING_SAMPLING_PARAMS  # noqa: E402

MODEL_PATH = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 4048
SAMPLING_PARAMS = THINKING_SAMPLING_PARAMS

STRONGREJECT_DIR = PROJECT_ROOT / "outputs/multivalue_dialect/strongreject"
PROBES = STRONGREJECT_DIR / "qwen3-8b" / "probes_dialects.jsonl"
OUTPUT_DIR = STRONGREJECT_DIR / "qwen3-8b"
BASELINE = PROJECT_ROOT / "outputs/strongreject/qwen3/baseline.jsonl"

CLASSIFY_SCRIPT = "scripts/strongreject/qwen3/classify.py"

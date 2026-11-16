"""Output paths for HarmBench standard runs that use Nepali prompt and probe."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/harmbench/standard_nepali/paths.py -> scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import HARMBENCH_DIR  # noqa: E402

STANDARD_NEPALI_DIR = HARMBENCH_DIR / "standard_nepali"
NEPALI_PROBES = STANDARD_NEPALI_DIR / "qwen3-8b" / "probes_nepali.jsonl"


def probe_infer_path(model: str) -> Path:
    return STANDARD_NEPALI_DIR / model / "probes_reasoning.jsonl"


def nepali_reasoning_path(model: str) -> Path:
    return STANDARD_NEPALI_DIR / model / "probes_nepali_reasoning.jsonl"


def nepali_bypass_path(model: str) -> Path:
    return STANDARD_NEPALI_DIR / model / "bypass_reasoning_deepseekv3-03.jsonl"

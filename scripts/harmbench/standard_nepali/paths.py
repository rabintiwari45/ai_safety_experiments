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
    return STANDARD_NEPALI_DIR / model / "bypass_nepali_reasoning_deepseekv3_03.jsonl"


def nepali_bypass_infer_path(model: str, mode: str) -> Path:
    suffix = {
        "open_think": "_infer.jsonl",
        "closed_think": "_closed_think_infer.jsonl",
        "no_think": "_no_think_infer.jsonl",
    }[mode]
    return nepali_bypass_path(model).with_name(
        f"bypass_nepali_reasoning_deepseekv3_03{suffix}"
    )

"""Paths and model id for HarmBench standard runs on Qwen3-4B."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/harmbench/standard/qwen3-4b/<file>.py -> scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import HARMBENCH_STANDARD_DIR  # noqa: E402

# Same thinking model family as Qwen3-8B. enable_thinking and <think> still apply.
MODEL_PATH = "Qwen/Qwen3-4B"

QWEN3_4B_DIR = HARMBENCH_STANDARD_DIR / "qwen3-4b"
BASELINE_JSONL = QWEN3_4B_DIR / "baseline.jsonl"

PROBES = QWEN3_4B_DIR / "probes.jsonl"
PROBE_INFER = QWEN3_4B_DIR / "probes_reasoning.jsonl"
BYPASS = QWEN3_4B_DIR / "bypass_reasoning_deepseekv3-03.jsonl"
BYPASS_INFER = QWEN3_4B_DIR / "bypass_reasoning_deepseekv3-03_infer.jsonl"
BYPASS_INFER_CLASSIFIED = (
    QWEN3_4B_DIR / "bypass_reasoning_deepseekv3-03_infer_classified.jsonl"
)
BYPASS_OUTPUT_PREFIX_INFER = (
    QWEN3_4B_DIR / "bypass_reasoning_deepseekv3-03_no_think_infer.jsonl"
)
BYPASS_OUTPUT_PREFIX_CLASSIFIED = (
    QWEN3_4B_DIR / "bypass_reasoning_deepseekv3-03_no_think_infer_classified.jsonl"
)
BYPASS_CLOSED_THINK_INFER = (
    QWEN3_4B_DIR / "bypass_reasoning_deepseekv3-03_closed_think_infer.jsonl"
)
BYPASS_CLOSED_THINK_CLASSIFIED = (
    QWEN3_4B_DIR / "bypass_reasoning_deepseekv3-03_closed_think_infer_classified.jsonl"
)

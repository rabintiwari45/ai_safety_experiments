"""Paths and model id for HarmBench standard runs on Qwen3-32B-FP8."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/harmbench/standard/qwen3-32b-fp8/<file>.py -> scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import HARMBENCH_STANDARD_DIR  # noqa: E402

# FP8 checkpoint of Qwen3-32B. Same chat template as the other Qwen3 runs:
# enable_thinking and an open <think> block still apply.
MODEL_PATH = "Qwen/Qwen3-32B-FP8"

QWEN3_32B_FP8_DIR = HARMBENCH_STANDARD_DIR / "qwen3-32b-fp8"
BASELINE_JSONL = QWEN3_32B_FP8_DIR / "baseline.jsonl"

PROBES = QWEN3_32B_FP8_DIR / "probes.jsonl"
PROBE_INFER = QWEN3_32B_FP8_DIR / "probes_reasoning.jsonl"
BYPASS = QWEN3_32B_FP8_DIR / "bypass_reasoning_deepseekv3-03.jsonl"
BYPASS_INFER = QWEN3_32B_FP8_DIR / "bypass_reasoning_deepseekv3-03_infer.jsonl"
BYPASS_INFER_CLASSIFIED = (
    QWEN3_32B_FP8_DIR / "bypass_reasoning_deepseekv3-03_infer_classified.jsonl"
)
BYPASS_OUTPUT_PREFIX_INFER = (
    QWEN3_32B_FP8_DIR / "bypass_reasoning_deepseekv3-03_no_think_infer.jsonl"
)
BYPASS_OUTPUT_PREFIX_CLASSIFIED = (
    QWEN3_32B_FP8_DIR / "bypass_reasoning_deepseekv3-03_no_think_infer_classified.jsonl"
)

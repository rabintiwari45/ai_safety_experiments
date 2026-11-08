"""Paths and model id for HarmBench contextual runs on Qwen3-4B without context.

The target model is shown the Behavior only. ContextString is kept on each
row so the official HarmBench contextual classifier can still score outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/harmbench/contextual_without_context/qwen3-4b/<file>.py -> scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import HARMBENCH_DIR  # noqa: E402

MODEL_PATH = "Qwen/Qwen3-4B"

QWEN3_4B_DIR = HARMBENCH_DIR / "contextual_without_context" / "qwen3-4b"

BEHAVIORS = QWEN3_4B_DIR / "harmbench_contextual_behaviors.jsonl"
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

"""Paths and sampling for HarmBench standard runs on Llama 3 8B Instruct."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/harmbench/standard/llama3/<file>.py -> scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import HARMBENCH_DIR, HARMBENCH_STANDARD_QWEN3_DIR  # noqa: E402

MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"

LLAMA3_DIR = HARMBENCH_DIR / "llama3"
STANDARD_PROBES = HARMBENCH_STANDARD_QWEN3_DIR / "probes.jsonl"
STANDARD_BYPASS = HARMBENCH_STANDARD_QWEN3_DIR / "bypass_reasoning_deepseekv3-03.jsonl"
BASELINE_JSONL = LLAMA3_DIR / "baseline.jsonl"

PROBE_INFER = LLAMA3_DIR / "probes_reasoning.jsonl"
BYPASS_INFER = LLAMA3_DIR / "bypass_reasoning_deepseekv3-03_infer.jsonl"
BYPASS_INFER_CLASSIFIED = (
    LLAMA3_DIR / "bypass_reasoning_deepseekv3-03_infer_classified.jsonl"
)
BYPASS_OUTPUT_PREFIX_INFER = (
    LLAMA3_DIR / "bypass_reasoning_deepseekv3-03_output_prefix_infer.jsonl"
)
BYPASS_OUTPUT_PREFIX_CLASSIFIED = (
    LLAMA3_DIR / "bypass_reasoning_deepseekv3-03_output_prefix_infer_classified.jsonl"
)

# Llama 3 Instruct defaults (generation_config.json on the model card).
LLAMA_SAMPLING_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.9,
    "max_new_tokens": 2048,
    "skip_special_tokens": False,
}


def chat_header(tokenizer, user_prompt: str) -> str:
    """Llama 3 chat header ending at the assistant generation prompt.

    Llama's template does not take Qwen's enable_thinking flag.
    """
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def split_think(continuation: str) -> tuple[str, str]:
    """Split a continuation on </think>.

    Llama 3 does not emit a think channel. When the close tag is absent,
    the continuation is the visible answer.
    """
    marker = "</think>"
    close = continuation.find(marker)
    if close == -1:
        return "", continuation.strip()
    reasoning = continuation[:close].strip()
    output = continuation[close + len(marker) :].strip()
    return reasoning, output

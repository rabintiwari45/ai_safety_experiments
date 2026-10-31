"""Paths and sampling for HarmBench standard runs on Gemma 4 E4B Instruct."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/harmbench/standard/gemma4/<file>.py -> scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import HARMBENCH_STANDARD_DIR, HARMBENCH_STANDARD_QWEN3_DIR  # noqa: E402

MODEL_PATH = "google/gemma-4-E4B-it"

GEMMA4_DIR = HARMBENCH_STANDARD_DIR / "gemma4"
STANDARD_PROBES = HARMBENCH_STANDARD_QWEN3_DIR / "probes.jsonl"
STANDARD_BYPASS = HARMBENCH_STANDARD_QWEN3_DIR / "bypass_reasoning_deepseekv3-03.jsonl"
BASELINE_JSONL = GEMMA4_DIR / "baseline.jsonl"

PROBE_INFER = GEMMA4_DIR / "probes_reasoning.jsonl"
BYPASS_INFER = GEMMA4_DIR / "bypass_reasoning_deepseekv3-03_infer.jsonl"
BYPASS_INFER_CLASSIFIED = (
    GEMMA4_DIR / "bypass_reasoning_deepseekv3-03_infer_classified.jsonl"
)
BYPASS_OUTPUT_PREFIX_INFER = (
    GEMMA4_DIR / "bypass_reasoning_deepseekv3-03_output_prefix_infer.jsonl"
)
BYPASS_OUTPUT_PREFIX_CLASSIFIED = (
    GEMMA4_DIR / "bypass_reasoning_deepseekv3-03_output_prefix_infer_classified.jsonl"
)

# Gemma 4 sampling from the model card.
GEMMA_SAMPLING_PARAMS = {
    "temperature": 1.0,
    "top_p": 0.95,
    "max_new_tokens": 2048,
    "skip_special_tokens": False,
}

# E4B thought channel. The close token is `<channel|>`, not `</think>`.
THOUGHT_OPEN = "<|channel>thought\n"
THOUGHT_CLOSE = "<channel|>"


def chat_header(tokenizer, user_prompt: str, *, enable_thinking: bool) -> str:
    """Gemma 4 chat header. enable_thinking inserts `<|think|>` in the system turn."""
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def join_open_thought(prefix: str) -> str:
    """Open the thought channel and leave `<channel|>` unclosed.

    E4B turns thinking on with `<|think|>`, but the generation prompt still
    ends at `<|turn>model` and does not open `<|channel>thought` by itself.
    """
    return THOUGHT_OPEN + prefix.rstrip()


def split_thought(continuation: str) -> tuple[str, str]:
    """Split a continuation on `<channel|>`.

    Text before the close stays in the thought channel. Text after it is the
    visible answer. If the model never closes the channel, the continuation
    is returned as the answer so the row is not stored with an empty output.
    """
    close = continuation.find(THOUGHT_CLOSE)
    if close == -1:
        text = continuation.strip()
        return text, text
    reasoning = continuation[:close].strip()
    output = continuation[close + len(THOUGHT_CLOSE) :].strip()
    return reasoning, output

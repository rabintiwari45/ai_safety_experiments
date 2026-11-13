"""Paths and Gemma 4 thought-channel helpers for StrongREJECT dialect probes."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/multivalue_dialect/strongreject/gemma4/<file>.py -> scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import PROJECT_ROOT  # noqa: E402

MODEL_PATH = "google/gemma-4-E4B-it"
MAX_NEW_TOKENS = 8048

STRONGREJECT_DIR = PROJECT_ROOT / "outputs/multivalue_dialect/strongreject"
PROBES = STRONGREJECT_DIR / "gemma4" / "probes_dialects.jsonl"
OUTPUT_DIR = STRONGREJECT_DIR / "gemma4"
BASELINE = PROJECT_ROOT / "outputs/strongreject/gemma4/baseline.jsonl"

CLASSIFY_SCRIPT = "scripts/strongreject/gemma4/classify.py"

# Gemma 4 sampling from the model card. max_new_tokens is replaced at run time.
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
    """Open the thought channel and leave `<channel|>` unclosed."""
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

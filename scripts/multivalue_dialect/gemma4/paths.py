"""Paths and sampling for HarmBench standard dialect probes on Gemma 4 E4B."""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/multivalue_dialect/gemma4/<file>.py -> scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import HARMBENCH_STANDARD_DIR, PROJECT_ROOT  # noqa: E402

MODEL_PATH = "google/gemma-4-E4B-it"
MAX_NEW_TOKENS = 8048

MULTIVALUE_STANDARD_DIR = (
    PROJECT_ROOT / "outputs/multivalue_dialect/harmbench/standard"
)
# Dialect text is shared. Qwen3-8B already holds probes_dialects.jsonl.
PROBES = MULTIVALUE_STANDARD_DIR / "qwen3-8b" / "probes_dialects.jsonl"
OUTPUT_DIR = MULTIVALUE_STANDARD_DIR / "gemma4"
BASELINE = HARMBENCH_STANDARD_DIR / "gemma4" / "baseline.jsonl"

DEFAULT_INFER = OUTPUT_DIR / "probes_infer_african_american_vernacular.jsonl"
DEFAULT_INFER_CLASSIFIED = (
    OUTPUT_DIR / "probes_infer_african_american_vernacular_classified.jsonl"
)
SINGAPORE_INFER = OUTPUT_DIR / "probes_infer_colloquial_singapore.jsonl"
SINGAPORE_INFER_CLASSIFIED = (
    OUTPUT_DIR / "probes_infer_colloquial_singapore_classified.jsonl"
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
    """Open the thought channel and leave `<channel|>` unclosed."""
    return THOUGHT_OPEN + prefix.rstrip()


def join_closed_thought(prefix: str) -> str:
    """Bypass inside a finished thought channel. Generation starts after `<channel|>`."""
    return f"{THOUGHT_OPEN}{prefix.rstrip()}\n{THOUGHT_CLOSE}"


def split_closed_thought(continuation: str) -> tuple[str, str]:
    """Answer starts immediately. Extra text before a new `<channel|>` is reasoning."""
    close = continuation.find(THOUGHT_CLOSE)
    if close == -1:
        return "", continuation.strip()
    extra = continuation[:close].strip()
    answer = continuation[close + len(THOUGHT_CLOSE) :].strip()
    return extra, answer


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

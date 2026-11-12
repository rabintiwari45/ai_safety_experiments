"""Qwen3-8B dialect prompt + dialect bypass, three think modes.

open_think    <think> stays open. The model continues the dialect bypass.
closed_think  <think>...</think> is closed. The model writes the answer.
no_think      enable_thinking=False. The dialect bypass is assistant text.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import configure_model_hub_env  # noqa: E402
from paths import BASELINE, MODEL_PATH, OUTPUT_DIR, SAMPLING_PARAMS  # noqa: E402

configure_model_hub_env(MODEL_PATH)

import bypass_infer_loop  # noqa: E402

CLASSIFY_SCRIPT = "scripts/multivalue_dialect/qwen3/classify.py"
MODEL_LABEL = "QWEN"


def join_open_think(prefix: str) -> str:
    # No </think>. No trailing newline after a finished-looking sentence,
    # or Qwen often closes thinking immediately.
    return f"<think>\n{prefix.rstrip()}"


def join_closed_think(prefix: str) -> str:
    """Bypass inside a finished think block. A blank line follows </think>."""
    return f"<think>\n{prefix.rstrip()}\n</think>\n\n"


def qwen_header(tokenizer, user_prompt: str, *, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_open(tokenizer, user_prompt: str, bypass: str) -> str:
    return qwen_header(tokenizer, user_prompt, enable_thinking=True) + join_open_think(
        bypass
    )


def build_closed(tokenizer, user_prompt: str, bypass: str) -> str:
    return qwen_header(
        tokenizer, user_prompt, enable_thinking=True
    ) + join_closed_think(bypass)


def build_no_think(tokenizer, user_prompt: str, bypass: str) -> str:
    return qwen_header(
        tokenizer, user_prompt, enable_thinking=False
    ) + bypass.rstrip()


def new_reasoning_from_continuation(continuation: str) -> str:
    close = continuation.find("</think>")
    if close == -1:
        return continuation
    return continuation[:close]


def split_continuation(continuation: str) -> tuple[str, str]:
    """Answer starts immediately. Extra text before a new </think> is reasoning."""
    close = continuation.find("</think>")
    if close == -1:
        return "", continuation.strip()
    extra = continuation[:close].strip()
    answer = continuation[close + len("</think>") :].strip()
    return extra, answer


def parse_open(continuation: str, bypass: str, parser) -> dict:
    stitched = join_open_think(bypass) + continuation
    reasoning_text, content = parser.parse_non_stream(stitched)
    return {
        "new_reasoning": new_reasoning_from_continuation(continuation).strip(),
        "reasoning": (reasoning_text or "").strip(),
        "output": (content or "").strip(),
        "raw_text": stitched,
        "new_output_suffix": "",
    }


def parse_closed(continuation: str, bypass: str, parser) -> dict:
    stitched = join_closed_think(bypass) + continuation
    reasoning_text, content = parser.parse_non_stream(stitched)
    extra_reasoning, answer = split_continuation(continuation)
    return {
        "new_reasoning": extra_reasoning,
        "reasoning": (reasoning_text or "").strip(),
        "output": (content or answer).strip(),
        "raw_text": stitched,
        "new_output_suffix": "",
    }


def parse_no_think(continuation: str, bypass: str, parser) -> dict:
    full_assistant = bypass + continuation
    reasoning_text, content = parser.parse_non_stream(full_assistant)
    suffix = continuation.strip()
    return {
        "new_reasoning": "",
        "reasoning": (reasoning_text or "").strip(),
        "output": (content or full_assistant).strip(),
        "raw_text": full_assistant,
        "new_output_suffix": suffix,
    }


def load_engine(model_path: str):
    import sglang as sgl
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    llm = sgl.Engine(
        model_path=model_path,
        reasoning_parser="qwen3",
        mem_fraction_static=0.85,
    )
    return llm, ReasoningParser("qwen3")


MODES = {
    "open_think": (
        build_open,
        parse_open,
        "BYPASS IN <think> (augmented, </think> not closed)",
        "Qwen3-8B continue from a dialect bypass (open <think>)",
    ),
    "closed_think": (
        build_closed,
        parse_closed,
        "BYPASS IN <think> (augmented, </think> closed)",
        "Qwen3-8B answer after a closed <think> dialect bypass",
    ),
    "no_think": (
        build_no_think,
        parse_no_think,
        "BYPASS PREFIX (assistant output, thinking off)",
        "Qwen3-8B continue a dialect bypass as assistant output (no think)",
    ),
}


def main_for(mode: str) -> None:
    build_continue_prompt, parse_generation, channel_banner, description = MODES[mode]
    args = bypass_infer_loop.parse_args(
        description=description,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        baseline=BASELINE,
        mode=mode,
    )
    bypass_infer_loop.run(
        args,
        sampling_params=SAMPLING_PARAMS,
        build_continue_prompt=build_continue_prompt,
        parse_generation=parse_generation,
        load_engine=load_engine,
        model_label=MODEL_LABEL,
        channel_banner=channel_banner,
        classify_script=CLASSIFY_SCRIPT,
    )

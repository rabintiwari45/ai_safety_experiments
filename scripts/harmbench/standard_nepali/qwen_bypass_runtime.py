"""Qwen Nepali prompt + Nepali bypass, three think modes.

open_think    <think> stays open. The model continues the Nepali bypass.
closed_think  <think>...</think> is closed. The model writes the answer.
no_think      enable_thinking=False. The Nepali bypass is assistant text.
"""

from __future__ import annotations

import sys
from pathlib import Path

NEPALI_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = NEPALI_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
sys.path.insert(0, str(NEPALI_DIR))

from common import THINKING_SAMPLING_PARAMS, configure_model_hub_env  # noqa: E402
from paths import STANDARD_NEPALI_DIR  # noqa: E402

import bypass_infer_loop  # noqa: E402

MODELS = {
    "qwen3-8b": {
        "model_path": "Qwen/Qwen3-8B",
        "label": "QWEN3-8B",
        "max_new_tokens": 12048,
    },
    "qwen3-4b": {
        "model_path": "Qwen/Qwen3-4B",
        "label": "QWEN3-4B",
        "max_new_tokens": 12048,
    },
}


def join_open_think(prefix: str) -> str:
    return f"<think>\n{prefix.rstrip()}"


def join_closed_think(prefix: str) -> str:
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
        "NEPALI BYPASS IN <think> (</think> not closed)",
        "continue from a Nepali bypass (open <think>)",
    ),
    "closed_think": (
        build_closed,
        parse_closed,
        "NEPALI BYPASS IN <think> (</think> closed)",
        "answer after a closed <think> Nepali bypass",
    ),
    "no_think": (
        build_no_think,
        parse_no_think,
        "NEPALI BYPASS PREFIX (assistant output, thinking off)",
        "continue a Nepali bypass as assistant output (no think)",
    ),
}


def main_for(mode: str, model_key: str) -> None:
    spec = MODELS[model_key]
    configure_model_hub_env(spec["model_path"])
    build_continue_prompt, parse_generation, channel_banner, description = MODES[mode]
    record_mode = (
        "bypass_output_prefix_empty_think" if mode == "no_think" else None
    )
    args = bypass_infer_loop.parse_args(
        description=f"{spec['label']} {description}",
        model_path=spec["model_path"],
        output_dir=STANDARD_NEPALI_DIR / model_key,
        mode=mode,
        max_new_tokens=spec["max_new_tokens"],
    )
    bypass_infer_loop.run(
        args,
        sampling_params=THINKING_SAMPLING_PARAMS,
        build_continue_prompt=build_continue_prompt,
        parse_generation=parse_generation,
        load_engine=load_engine,
        model_label=spec["label"],
        channel_banner=channel_banner,
        record_mode=record_mode,
        # Off for now. The per-token scan runs on the CPU and stalls decode.
        stop_on_repetition=False,
    )

"""Gemma 4 Nepali prompt + Nepali bypass, three think modes.

open_think    <|channel>thought stays open. The model continues the Nepali bypass.
closed_think  the thought channel is closed with <channel|>. The model writes the answer.
no_think      enable_thinking=False. The Nepali bypass is assistant text.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

NEPALI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = NEPALI_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
sys.path.insert(0, str(NEPALI_DIR))

from common import configure_model_hub_env  # noqa: E402
from paths import STANDARD_NEPALI_DIR  # noqa: E402

_GEMMA_PATHS = SCRIPTS_DIR / "harmbench" / "standard" / "gemma4" / "paths.py"
_spec = importlib.util.spec_from_file_location("gemma4_standard_paths", _GEMMA_PATHS)
_gemma = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_gemma)

configure_model_hub_env(_gemma.MODEL_PATH)

import bypass_infer_loop  # noqa: E402

MODEL_LABEL = "GEMMA4"


def build_open(tokenizer, user_prompt: str, bypass: str) -> str:
    return _gemma.chat_header(
        tokenizer, user_prompt, enable_thinking=True
    ) + _gemma.join_open_thought(bypass)


def build_closed(tokenizer, user_prompt: str, bypass: str) -> str:
    return _gemma.chat_header(
        tokenizer, user_prompt, enable_thinking=True
    ) + _gemma.join_closed_thought(bypass)


def build_no_think(tokenizer, user_prompt: str, bypass: str) -> str:
    return _gemma.chat_header(
        tokenizer, user_prompt, enable_thinking=False
    ) + bypass.rstrip()


def parse_open(continuation: str, bypass: str, _parser) -> dict:
    new_reasoning, output = _gemma.split_thought(continuation)
    return {
        "new_reasoning": new_reasoning,
        "reasoning": new_reasoning,
        "output": output,
        "raw_text": _gemma.join_open_thought(bypass) + continuation,
        "new_output_suffix": "",
    }


def parse_closed(continuation: str, bypass: str, _parser) -> dict:
    new_reasoning, output = _gemma.split_closed_thought(continuation)
    return {
        "new_reasoning": new_reasoning,
        "reasoning": new_reasoning,
        "output": output,
        "raw_text": _gemma.join_closed_thought(bypass) + continuation,
        "new_output_suffix": "",
    }


def parse_no_think(continuation: str, bypass: str, _parser) -> dict:
    suffix = continuation.strip()
    output = (bypass + continuation).strip()
    return {
        "new_reasoning": "",
        "reasoning": "",
        "output": output,
        "raw_text": output,
        "new_output_suffix": suffix,
    }


def load_engine(model_path: str):
    import sglang as sgl

    llm = sgl.Engine(model_path=model_path, mem_fraction_static=0.85)
    return llm, None


MODES = {
    "open_think": (
        build_open,
        parse_open,
        "NEPALI BYPASS IN <|channel>thought (<channel|> not closed)",
        "continue from a Nepali bypass (open thought channel)",
        "bypass_open_think",
    ),
    "closed_think": (
        build_closed,
        parse_closed,
        "NEPALI BYPASS IN <|channel>thought (<channel|> closed)",
        "answer after a closed thought-channel Nepali bypass",
        "bypass_closed_think",
    ),
    "no_think": (
        build_no_think,
        parse_no_think,
        "NEPALI BYPASS PREFIX (assistant output, thinking off)",
        "continue a Nepali bypass as assistant output (no think)",
        "bypass_output_prefix",
    ),
}


def main_for(mode: str) -> None:
    build_continue_prompt, parse_generation, channel_banner, description, record_mode = (
        MODES[mode]
    )
    args = bypass_infer_loop.parse_args(
        description=f"Gemma 4 E4B {description}",
        model_path=_gemma.MODEL_PATH,
        output_dir=STANDARD_NEPALI_DIR / "gemma4",
        mode=mode,
        max_new_tokens=12048,
    )
    bypass_infer_loop.run(
        args,
        sampling_params=_gemma.GEMMA_SAMPLING_PARAMS,
        build_continue_prompt=build_continue_prompt,
        parse_generation=parse_generation,
        load_engine=load_engine,
        model_label=MODEL_LABEL,
        channel_banner=channel_banner,
        record_mode=record_mode,
        stop_on_repetition=True,
    )

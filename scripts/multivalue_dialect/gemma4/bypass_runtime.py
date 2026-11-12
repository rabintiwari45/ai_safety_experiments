"""Gemma 4 E4B dialect prompt + dialect bypass, three think modes.

open_think    <|channel>thought stays open. The model continues the dialect bypass.
closed_think  the thought channel is closed with <channel|>. The model writes the answer.
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
from paths import (  # noqa: E402
    BASELINE,
    GEMMA_SAMPLING_PARAMS,
    MODEL_PATH,
    OUTPUT_DIR,
    chat_header,
    join_closed_thought,
    join_open_thought,
    split_closed_thought,
    split_thought,
)

configure_model_hub_env(MODEL_PATH)

import bypass_infer_loop  # noqa: E402

CLASSIFY_SCRIPT = "scripts/multivalue_dialect/gemma4/classify.py"
MODEL_LABEL = "GEMMA"


def build_open(tokenizer, user_prompt: str, bypass: str) -> str:
    return chat_header(
        tokenizer, user_prompt, enable_thinking=True
    ) + join_open_thought(bypass)


def build_closed(tokenizer, user_prompt: str, bypass: str) -> str:
    return chat_header(
        tokenizer, user_prompt, enable_thinking=True
    ) + join_closed_thought(bypass)


def build_no_think(tokenizer, user_prompt: str, bypass: str) -> str:
    return chat_header(
        tokenizer, user_prompt, enable_thinking=False
    ) + bypass.rstrip()


def parse_open(continuation: str, bypass: str, _parser) -> dict:
    new_reasoning, output = split_thought(continuation)
    return {
        "new_reasoning": new_reasoning,
        "reasoning": new_reasoning,
        "output": output,
        "raw_text": join_open_thought(bypass) + continuation,
        "new_output_suffix": "",
    }


def parse_closed(continuation: str, bypass: str, _parser) -> dict:
    new_reasoning, output = split_closed_thought(continuation)
    return {
        "new_reasoning": new_reasoning,
        "reasoning": new_reasoning,
        "output": output,
        "raw_text": join_closed_thought(bypass) + continuation,
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

    llm = sgl.Engine(
        model_path=model_path,
        mem_fraction_static=0.85,
    )
    return llm, None


MODES = {
    "open_think": (
        build_open,
        parse_open,
        "BYPASS IN <|channel>thought (augmented, <channel|> not closed)",
        "Gemma 4 E4B continue from a dialect bypass (open thought channel)",
    ),
    "closed_think": (
        build_closed,
        parse_closed,
        "BYPASS IN <|channel>thought (augmented, <channel|> closed)",
        "Gemma 4 E4B answer after a closed thought-channel dialect bypass",
    ),
    "no_think": (
        build_no_think,
        parse_no_think,
        "BYPASS PREFIX (assistant output, thinking off)",
        "Gemma 4 E4B continue a dialect bypass as assistant output (no think)",
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
    record_mode = "bypass_output_prefix" if mode == "no_think" else None
    bypass_infer_loop.run(
        args,
        sampling_params=GEMMA_SAMPLING_PARAMS,
        build_continue_prompt=build_continue_prompt,
        parse_generation=parse_generation,
        load_engine=load_engine,
        model_label=MODEL_LABEL,
        channel_banner=channel_banner,
        classify_script=CLASSIFY_SCRIPT,
        record_mode=record_mode,
    )

"""Gemma 4 E4B open-thought continuation for one dialect probe file."""

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
    MAX_NEW_TOKENS,
    MODEL_PATH,
    OUTPUT_DIR,
    PROBES,
    chat_header,
    join_open_thought,
    split_thought,
)

configure_model_hub_env(MODEL_PATH)

import infer_loop  # noqa: E402

CLASSIFY_SCRIPT = "scripts/multivalue_dialect/gemma4/classify.py"


def build_continue_prompt(tokenizer, user_prompt: str, probe: str) -> str:
    return chat_header(
        tokenizer, user_prompt, enable_thinking=True
    ) + join_open_thought(probe)


def parse_generation(continuation: str, probe: str, _parser) -> tuple[str, str, str, str]:
    new_reasoning, output = split_thought(continuation)
    stitched = join_open_thought(probe) + continuation
    return new_reasoning, new_reasoning, output, stitched


def load_engine(model_path: str):
    import sglang as sgl

    llm = sgl.Engine(
        model_path=model_path,
        mem_fraction_static=0.85,
    )
    return llm, None


def main_for(dialect: str) -> None:
    args = infer_loop.parse_args(
        description=(
            f"Gemma 4 E4B continue from {dialect} dialect probes "
            "(open thought channel)"
        ),
        model_path=MODEL_PATH,
        probes=PROBES,
        baseline=BASELINE,
        max_new_tokens=MAX_NEW_TOKENS,
        dialect=dialect,
    )
    infer_loop.run(
        args,
        output_dir=OUTPUT_DIR,
        sampling_params=GEMMA_SAMPLING_PARAMS,
        build_continue_prompt=build_continue_prompt,
        parse_generation=parse_generation,
        load_engine=load_engine,
        model_label="GEMMA",
        probe_channel="<|channel>thought",
        classify_script=CLASSIFY_SCRIPT,
    )

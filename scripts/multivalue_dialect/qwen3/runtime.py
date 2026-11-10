"""Qwen3-8B open-<think> continuation for one dialect probe file."""

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
    MAX_NEW_TOKENS,
    MODEL_PATH,
    OUTPUT_DIR,
    PROBES,
    SAMPLING_PARAMS,
)

configure_model_hub_env(MODEL_PATH)

import infer_loop  # noqa: E402

CLASSIFY_SCRIPT = "scripts/multivalue_dialect/qwen3/classify.py"


def join_open_think(probe: str) -> str:
    # No </think>. No trailing newline after a finished-looking sentence,
    # or Qwen often closes thinking immediately.
    return f"<think>\n{probe.rstrip()}"


def build_continue_prompt(tokenizer, user_prompt: str, probe: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + join_open_think(probe)


def new_reasoning_from_continuation(continuation: str) -> str:
    close = continuation.find("</think>")
    if close == -1:
        return continuation
    return continuation[:close]


def parse_generation(continuation: str, probe: str, parser) -> tuple[str, str, str, str]:
    stitched = join_open_think(probe) + continuation
    reasoning_text, content = parser.parse_non_stream(stitched)
    output = (content or "").strip()
    new_reasoning = new_reasoning_from_continuation(continuation).strip()
    return new_reasoning, (reasoning_text or "").strip(), output, stitched


def load_engine(model_path: str):
    import sglang as sgl
    from sglang.srt.parser.reasoning_parser import ReasoningParser

    llm = sgl.Engine(
        model_path=model_path,
        reasoning_parser="qwen3",
        mem_fraction_static=0.85,
    )
    return llm, ReasoningParser("qwen3")


def main_for(dialect: str) -> None:
    args = infer_loop.parse_args(
        description=(
            f"Qwen3-8B continue from {dialect} dialect probes (open <think>)"
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
        sampling_params=SAMPLING_PARAMS,
        build_continue_prompt=build_continue_prompt,
        parse_generation=parse_generation,
        load_engine=load_engine,
        model_label="QWEN",
        probe_channel="<think>",
        classify_script=CLASSIFY_SCRIPT,
    )

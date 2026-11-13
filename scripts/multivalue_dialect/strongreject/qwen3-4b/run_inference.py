"""
Run Qwen3-4B on one StrongREJECT dialect probe file.

User message: prompt_<dialect>
Open <think> prefix: probe_<dialect>. </think> is not closed.

Reads outputs/multivalue_dialect/strongreject/qwen3-4b/probes_dialects.jsonl.
Writes probes_dialects_reasoning_african_american.jsonl or
probes_dialects_reasoning_singapore.jsonl in that same directory.

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/strongreject/qwen3-4b/run_inference.py \\
        --dialect african_american_vernacular --dry-run --limit 1
    python scripts/multivalue_dialect/strongreject/qwen3-4b/run_inference.py \\
        --dialect colloquial_singapore --limit 313
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from paths import (  # noqa: E402
    BASELINE,
    CLASSIFY_SCRIPT,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    OUTPUT_DIR,
    PROBES,
    SAMPLING_PARAMS,
)

SCRIPTS_DIR = HERE.parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
from common import configure_model_hub_env  # noqa: E402

configure_model_hub_env(MODEL_PATH)

import infer  # noqa: E402


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


def main() -> None:
    args = infer.parse_args(
        description="Qwen3-4B StrongREJECT continue from dialect probes (open <think>)",
        model_path=MODEL_PATH,
        probes=PROBES,
        baseline=BASELINE,
        output_dir=OUTPUT_DIR,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    infer.run_dialect(
        args,
        sampling_params=SAMPLING_PARAMS,
        build_continue_prompt=build_continue_prompt,
        parse_generation=parse_generation,
        load_engine=load_engine,
        model_label="QWEN",
        probe_channel="<think>",
        classify_script=CLASSIFY_SCRIPT,
    )


if __name__ == "__main__":
    main()

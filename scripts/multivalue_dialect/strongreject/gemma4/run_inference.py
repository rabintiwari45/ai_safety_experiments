"""
Run Gemma 4 E4B on one StrongREJECT dialect probe file.

User message: prompt_<dialect>
Open thought prefix: probe_<dialect>. <channel|> is not closed.

Reads outputs/multivalue_dialect/strongreject/gemma4/probes_dialects.jsonl.
Writes probes_dialects_reasoning_african_american.jsonl or
probes_dialects_reasoning_singapore.jsonl in that same directory.

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/strongreject/gemma4/run_inference.py \\
        --dialect african_american_vernacular --dry-run --limit 1
    python scripts/multivalue_dialect/strongreject/gemma4/run_inference.py \\
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
    GEMMA_SAMPLING_PARAMS,
    MAX_NEW_TOKENS,
    MODEL_PATH,
    OUTPUT_DIR,
    PROBES,
    chat_header,
    join_open_thought,
    split_thought,
)

SCRIPTS_DIR = HERE.parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
from common import configure_model_hub_env  # noqa: E402

configure_model_hub_env(MODEL_PATH)

import infer  # noqa: E402


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


def main() -> None:
    args = infer.parse_args(
        description=(
            "Gemma 4 E4B StrongREJECT continue from dialect probes "
            "(open thought channel)"
        ),
        model_path=MODEL_PATH,
        probes=PROBES,
        baseline=BASELINE,
        output_dir=OUTPUT_DIR,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    infer.run_dialect(
        args,
        sampling_params=GEMMA_SAMPLING_PARAMS,
        build_continue_prompt=build_continue_prompt,
        parse_generation=parse_generation,
        load_engine=load_engine,
        model_label="GEMMA",
        probe_channel="<|channel>thought",
        classify_script=CLASSIFY_SCRIPT,
    )


if __name__ == "__main__":
    main()

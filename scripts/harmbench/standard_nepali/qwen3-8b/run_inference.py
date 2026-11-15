"""
Run Qwen3-8B from the Nepali HarmBench prompt and probe.

The Nepali probe is prefilled inside an open <think> tag. </think> is not closed.
The model continues thinking and writes the final answer.

Reads outputs/harmbench/standard_nepali/qwen3-8b/probes_nepali.jsonl.
Writes outputs/harmbench/standard_nepali/qwen3-8b/probes_reasoning.jsonl.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/standard_nepali/qwen3-8b/run_inference.py --dry-run --limit 1
    python scripts/harmbench/standard_nepali/qwen3-8b/run_inference.py --limit 1
    python scripts/harmbench/standard_nepali/qwen3-8b/run_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

NEPALI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = NEPALI_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(NEPALI_DIR))

from common import MODEL_PATH, THINKING_SAMPLING_PARAMS  # noqa: E402
from paths import probe_infer_path  # noqa: E402
from probe_inference import (  # noqa: E402
    join_open_think,
    qwen_parse,
    run_probe_inference,
)


def build_continue_prompt(tokenizer, user_prompt: str, probe: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + join_open_think(probe)


def main() -> None:
    run_probe_inference(
        description="Qwen3-8B continue from Nepali probes (open <think>)",
        model_path=MODEL_PATH,
        default_output=probe_infer_path("qwen3-8b"),
        sampling_base=THINKING_SAMPLING_PARAMS,
        default_max_new_tokens=12048,
        build_continue_prompt=build_continue_prompt,
        parse_generation=qwen_parse,
        engine_kwargs={"reasoning_parser": "qwen3", "mem_fraction_static": 0.85},
        parser_name="qwen3",
        full_prompt_title="QWEN3-8B",
        think_banner="NEPALI PROBE IN <think> (</think> not closed)",
    )


if __name__ == "__main__":
    main()

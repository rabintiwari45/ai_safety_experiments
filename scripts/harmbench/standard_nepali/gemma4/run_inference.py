"""
Run Gemma 4 E4B from the Nepali HarmBench prompt and probe.

Same condition as the English Gemma run: enable_thinking=True, then the
Nepali probe is prefilled inside an open thought channel (`<|channel>thought`)
and `<channel|>` is not closed.

Reads outputs/harmbench/standard_nepali/qwen3-8b/probes_nepali.jsonl.
Writes outputs/harmbench/standard_nepali/gemma4/probes_reasoning.jsonl.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/standard_nepali/gemma4/run_inference.py --dry-run --limit 1
    python scripts/harmbench/standard_nepali/gemma4/run_inference.py --limit 1
    python scripts/harmbench/standard_nepali/gemma4/run_inference.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

NEPALI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = NEPALI_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(NEPALI_DIR))

from paths import probe_infer_path  # noqa: E402
from probe_inference import run_probe_inference  # noqa: E402

_GEMMA_PATHS = (
    SCRIPTS_DIR / "harmbench" / "standard" / "gemma4" / "paths.py"
)
_spec = importlib.util.spec_from_file_location("gemma4_standard_paths", _GEMMA_PATHS)
_gemma = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_gemma)


def build_continue_prompt(tokenizer, user_prompt: str, probe: str) -> str:
    return _gemma.chat_header(
        tokenizer, user_prompt, enable_thinking=True
    ) + _gemma.join_open_thought(probe)


def parse_generation(continuation: str, probe: str, reasoning_parser) -> dict:
    new_reasoning, output = _gemma.split_thought(continuation)
    return {
        "new_reasoning": new_reasoning,
        "reasoning": new_reasoning,
        "output": output,
        "raw_text": _gemma.join_open_thought(probe) + continuation,
        "generated_text": continuation,
    }


def main() -> None:
    run_probe_inference(
        description="Gemma 4 E4B continue from Nepali probes (open thought channel)",
        model_path=_gemma.MODEL_PATH,
        default_output=probe_infer_path("gemma4"),
        sampling_base=_gemma.GEMMA_SAMPLING_PARAMS,
        default_max_new_tokens=8048,
        build_continue_prompt=build_continue_prompt,
        parse_generation=parse_generation,
        engine_kwargs={"mem_fraction_static": 0.85},
        parser_name=None,
        full_prompt_title="GEMMA4",
        think_banner="NEPALI PROBE IN <|channel>thought (<channel|> not closed)",
    )


if __name__ == "__main__":
    main()

"""
Qwen3-4B open-think bypass on a dialect prompt and dialect bypass trace.

User message: prompt_<dialect>
Open <think> prefix: bypass_reasoning_<dialect>, with the research line spliced in.
The English behavior stays in "prompt" for HarmBench classification.

Reads:
  outputs/multivalue_dialect/harmbench/standard/qwen3-4b/bypass_reasoning_deepseekv3-03_<dialect>_dialect.jsonl
Writes:
  outputs/multivalue_dialect/harmbench/standard/qwen3-4b/bypass_reasoning_deepseekv3-03_<dialect>_dialect_infer.jsonl

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/qwen3-4b/run_bypass_inference.py --dialect african_american_vernacular --dry-run --limit 1
    python scripts/multivalue_dialect/qwen3-4b/run_bypass_inference.py --dialect colloquial_singapore --limit 1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bypass_runtime import main_for  # noqa: E402

if __name__ == "__main__":
    main_for("open_think")

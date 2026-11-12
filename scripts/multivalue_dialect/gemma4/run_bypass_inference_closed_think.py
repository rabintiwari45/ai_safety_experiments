"""
Gemma 4 E4B closed-think bypass on a dialect prompt and dialect bypass trace.

User message: prompt_<dialect>
Closed thought channel: bypass_reasoning_<dialect>, with the research line
spliced in. Generation starts after <channel|>.

Reads:
  outputs/multivalue_dialect/harmbench/standard/gemma4/bypass_reasoning_deepseekv3-03_<dialect>_dialect.jsonl
Writes:
  outputs/multivalue_dialect/harmbench/standard/gemma4/bypass_reasoning_deepseekv3-03_<dialect>_dialect_closed_think_infer.jsonl

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/gemma4/run_bypass_inference_closed_think.py --dialect african_american_vernacular --dry-run --limit 1
    python scripts/multivalue_dialect/gemma4/run_bypass_inference_closed_think.py --dialect colloquial_singapore --limit 1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bypass_runtime import main_for  # noqa: E402

if __name__ == "__main__":
    main_for("closed_think")

"""
Qwen3-4B no-think bypass on a dialect prompt and dialect bypass trace.

User message: prompt_<dialect>
enable_thinking=False. The dialect bypass is prefilled as assistant text
after the empty think block the chat template inserts.

Reads:
  outputs/multivalue_dialect/harmbench/standard/qwen3-4b/bypass_reasoning_deepseekv3-03_<dialect>_dialect.jsonl
Writes:
  outputs/multivalue_dialect/harmbench/standard/qwen3-4b/bypass_reasoning_deepseekv3-03_<dialect>_dialect_no_think_infer.jsonl

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/qwen3-4b/run_bypass_inference_output_prefix.py --dialect african_american_vernacular --dry-run --limit 1
    python scripts/multivalue_dialect/qwen3-4b/run_bypass_inference_output_prefix.py --dialect colloquial_singapore --limit 1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bypass_runtime import main_for  # noqa: E402

if __name__ == "__main__":
    main_for("no_think")

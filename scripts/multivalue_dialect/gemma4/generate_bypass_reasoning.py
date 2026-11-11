"""
Rewrite Gemma 4 dialect reasoning with DeepSeek.

Reads the dialect prompt and probe (prompt_<dialect>, probe_<dialect>) from
probes_dialects_reasoning_<dialect>.jsonl and writes
bypass_reasoning_deepseekv3-03_<dialect>.jsonl.

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/gemma4/generate_bypass_reasoning.py --dialect colloquial_singapore --dry-run
    python scripts/multivalue_dialect/gemma4/generate_bypass_reasoning.py --dialect african_american_vernacular --limit 1 --print-prompt
    python scripts/multivalue_dialect/gemma4/generate_bypass_reasoning.py --dialect colloquial_singapore
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from generate_bypass_common import main  # noqa: E402
from paths import OUTPUT_DIR  # noqa: E402


if __name__ == "__main__":
    main(output_dir=OUTPUT_DIR, model_label="Gemma4")

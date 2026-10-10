"""Summarize a classified HarmBench standard JSONL.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/standard/summarize.py --input outputs/harmbench/standard/bypass_reasoning_deepseekv3-03_infer_classified.jsonl
    python scripts/harmbench/standard/summarize.py --input path/to/classified.jsonl --output path/to/summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))

from common import PROJECT_ROOT  # noqa: E402
import summarize_classified as sm  # noqa: E402

DEFAULT = (
    PROJECT_ROOT
    / "outputs"
    / "harmbench"
    / "standard"
    / "bypass_reasoning_deepseekv3-03_infer_classified.jsonl"
)

if __name__ == "__main__":
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", str(DEFAULT)])
    sm.main()

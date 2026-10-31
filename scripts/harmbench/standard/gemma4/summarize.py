"""Summarize a classified Gemma 4 HarmBench standard JSONL.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/standard/gemma4/summarize.py
    python scripts/harmbench/standard/gemma4/summarize.py --input outputs/harmbench/standard/gemma4/bypass_reasoning_deepseekv3-03_infer_classified.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import BYPASS_INFER_CLASSIFIED  # noqa: E402
import summarize_classified as sm  # noqa: E402

if __name__ == "__main__":
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", str(BYPASS_INFER_CLASSIFIED)])
    sm.main()

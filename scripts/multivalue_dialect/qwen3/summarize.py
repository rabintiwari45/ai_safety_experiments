"""Summarize a classified Qwen3-8B dialect-probe JSONL.

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/qwen3/summarize.py
    python scripts/multivalue_dialect/qwen3/summarize.py --input outputs/multivalue_dialect/harmbench/standard/qwen3-8b/probes_infer_colloquial_singapore_classified.jsonl
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paths import DEFAULT_INFER_CLASSIFIED  # noqa: E402
import summarize_classified as sm  # noqa: E402

if __name__ == "__main__":
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", str(DEFAULT_INFER_CLASSIFIED)])
    sm.main()

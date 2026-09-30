"""Summarize a classified dynamic intervention JSONL."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from common import DYNAMIC_DIR  # noqa: E402
import summarize_classified as sm  # noqa: E402

DEFAULT = DYNAMIC_DIR / "harmbench_standard_qwen3_8b_dynamic_classified.jsonl"

if __name__ == "__main__":
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", str(DEFAULT)])
    sm.main()

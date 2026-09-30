"""Classify Nepali dynamic-run JSONL."""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from common import TRANSLATION_RUN_JSONL  # noqa: E402
import classify_harmbench as cls  # noqa: E402

if __name__ == "__main__":
    if "--input" not in sys.argv:
        sys.argv.extend(["--input", str(TRANSLATION_RUN_JSONL)])
    cls.main()

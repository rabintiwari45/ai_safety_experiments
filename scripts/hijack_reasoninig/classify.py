"""
Classify hijack bypass-run outputs with GPT (HarmBench yes/no classifier).

Labels each row's final `output` against the HarmBench `prompt` (behavior).

Usage (from harmbench/):
    export OPENAI_API_KEY=...
    python scripts/hijack_reasoninig/classify.py
    python scripts/hijack_reasoninig/classify.py --limit 5
    python scripts/hijack_reasoninig/classify.py --input outputs/hijack_reasoninig/hijack_reasoninig_bypass_run.jsonl

Default classifier: gpt-5.6-luna (override with --model or OPENAI_MODEL).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))

from common import (  # noqa: E402
    HIJACK_CLASSIFIER_MODEL,
    HIJACK_REASONING_BYPASS_RUN,
    HIJACK_REASONING_BYPASS_RUN_CLASSIFIED,
)
import classify_harmbench as cls  # noqa: E402

def _argv_has_model_flag(argv: list[str]) -> bool:
    return any(a == "--model" or a.startswith("--model=") for a in argv)


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--input" not in argv:
        argv = ["--input", str(HIJACK_REASONING_BYPASS_RUN), *argv]
    if "--output" not in argv:
        argv = ["--output", str(HIJACK_REASONING_BYPASS_RUN_CLASSIFIED), *argv]
    if not _argv_has_model_flag(argv):
        model = os.environ.get("OPENAI_MODEL", HIJACK_CLASSIFIER_MODEL)
        argv = ["--model", model, *argv]
    sys.argv = [sys.argv[0], *argv]
    cls.main()

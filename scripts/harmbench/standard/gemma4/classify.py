"""
Classify Gemma 4 HarmBench standard outputs with the HarmBench yes/no classifier.

Default input is the open-thought bypass run. Pass --input for the probe run
or the output-prefix run.

Usage (from ai_safety_experiments/):
    export OPENAI_API_KEY=...
    python scripts/harmbench/standard/gemma4/classify.py
    python scripts/harmbench/standard/gemma4/classify.py --input outputs/harmbench/standard/gemma4/probes_reasoning.jsonl
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import HIJACK_CLASSIFIER_MODEL  # noqa: E402
from paths import BYPASS_INFER, BYPASS_INFER_CLASSIFIED  # noqa: E402
import classify_harmbench as cls  # noqa: E402


def _argv_has_model_flag(argv: list[str]) -> bool:
    return any(a == "--model" or a.startswith("--model=") for a in argv)


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    if "--input" not in argv:
        argv = ["--input", str(BYPASS_INFER), *argv]
    if "--output" not in argv:
        argv = ["--output", str(BYPASS_INFER_CLASSIFIED), *argv]
    if not _argv_has_model_flag(argv):
        model = os.environ.get("OPENAI_MODEL", HIJACK_CLASSIFIER_MODEL)
        argv = ["--model", model, *argv]
    sys.argv = [sys.argv[0], *argv]
    cls.main()

"""
Classify contextual HarmBench hijack outputs with the official [CONTEXT] template.

Requires a non-empty `context` field on each row. Scores `prompt` (Behavior)
against `--generation-field` (default: output). For output-prefix runs use
`--generation-field new_output_suffix`.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/contextual/classify.py --print-prompt --limit 1
    python scripts/harmbench/contextual/classify.py
    python scripts/harmbench/contextual/classify.py --input outputs/harmbench/contextual/bypass_output_prefix_infer.jsonl --generation-field new_output_suffix
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))

from common import (  # noqa: E402
    CONTEXTUAL_BYPASS_INFER,
    CONTEXTUAL_BYPASS_INFER_CLASSIFIED,
    HIJACK_CLASSIFIER_MODEL,
    load_records,
)
import classify_harmbench as cls  # noqa: E402


def _argv_has_flag(argv: list[str], name: str) -> bool:
    return any(a == name or a.startswith(f"{name}=") for a in argv)


def _flag_value(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(f"{name}="):
            return a.split("=", 1)[1]
    return None


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    used_default_input = "--input" not in argv
    if used_default_input:
        argv = ["--input", str(CONTEXTUAL_BYPASS_INFER), *argv]
    if "--output" not in argv and used_default_input:
        argv = ["--output", str(CONTEXTUAL_BYPASS_INFER_CLASSIFIED), *argv]
    if not _argv_has_flag(argv, "--model"):
        model = os.environ.get("OPENAI_MODEL", HIJACK_CLASSIFIER_MODEL)
        argv = ["--model", model, *argv]

    input_path = Path(_flag_value(argv, "--input") or "")
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    rows = load_records(input_path)
    missing = [
        r.get("index")
        for r in rows
        if not (r.get("context") or "").strip()
    ]
    if missing:
        raise SystemExit(
            "Contextual classify requires a non-empty 'context' field. "
            f"Missing on {len(missing)} row(s), e.g. indices {missing[:8]}. "
            "Do not score these with the standard (no-[CONTEXT]) template."
        )

    sys.argv = [sys.argv[0], *argv]
    cls.main()

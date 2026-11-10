"""
Classify Gemma 4 dialect-probe outputs with the HarmBench yes/no classifier.

Default input is the African American Vernacular run. Pass --input for
Colloquial Singapore.

Usage (from ai_safety_experiments/):
    export OPENAI_API_KEY=...
    python scripts/multivalue_dialect/gemma4/classify.py
    python scripts/multivalue_dialect/gemma4/classify.py --input outputs/multivalue_dialect/harmbench/standard/gemma4/probes_infer_colloquial_singapore.jsonl
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import HIJACK_CLASSIFIER_MODEL  # noqa: E402
from paths import DEFAULT_INFER, DEFAULT_INFER_CLASSIFIED  # noqa: E402
import classify_harmbench as cls  # noqa: E402


def _argv_has_flag(argv: list[str], name: str) -> bool:
    return any(a == name or a.startswith(f"{name}=") for a in argv)


def _pop_flag(argv: list[str], name: str) -> tuple[list[str], str | None]:
    kept: list[str] = []
    value: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == name or arg.startswith(f"{name}="):
            if value is not None:
                raise SystemExit(f"Pass {name} only once.")
            if arg == name:
                if i + 1 >= len(argv):
                    raise SystemExit(f"{name} needs a value.")
                value = argv[i + 1]
                i += 2
                continue
            value = arg.split("=", 1)[1]
            i += 1
            continue
        kept.append(arg)
        i += 1
    return kept, value


if __name__ == "__main__":
    argv = list(sys.argv[1:])
    argv, key = _pop_flag(argv, "--key")
    argv, generation_field = _pop_flag(argv, "--generation-field")
    if key and generation_field and key != generation_field:
        raise SystemExit(
            f"--key {key!r} and --generation-field {generation_field!r} disagree."
        )
    field = key or generation_field or "output"
    if not field.strip():
        raise SystemExit("--key needs a non-empty JSON field name.")
    argv = ["--generation-field", field, *argv]

    used_default_input = not _argv_has_flag(argv, "--input")
    if used_default_input:
        argv = ["--input", str(DEFAULT_INFER), *argv]
    if not _argv_has_flag(argv, "--output"):
        if used_default_input and field == "output":
            output = DEFAULT_INFER_CLASSIFIED
        else:
            _, input_path = _pop_flag(argv, "--input")
            stem = Path(input_path or "").stem
            suffix = "" if field == "output" else f"_{field}"
            output = DEFAULT_INFER_CLASSIFIED.with_name(
                f"{stem}{suffix}_classified.jsonl"
            )
        argv = ["--output", str(output), *argv]
    if not _argv_has_flag(argv, "--model"):
        model = os.environ.get("OPENAI_MODEL", HIJACK_CLASSIFIER_MODEL)
        argv = ["--model", model, *argv]
    if "--help" not in argv and "-h" not in argv:
        print(f"Scoring JSON field {field!r}", flush=True)
    sys.argv = [sys.argv[0], *argv]
    cls.main()

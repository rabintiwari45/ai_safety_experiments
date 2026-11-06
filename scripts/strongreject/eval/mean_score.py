"""Print the mean StrongREJECT score in a scored JSONL.

Reads the file written by evaluate.py. The mean uses finite
strongreject_score values only. null and the string "nan" are left out.
When several lines share an index, the last line is the one that counts,
matching evaluate.py.

Usage (from ai_safety_experiments/):
    python scripts/strongreject/eval/mean_score.py \
        --input outputs/strongreject/qwen3/probes_reasoning_strongreject_rubric.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean strongreject_score for a scored StrongREJECT JSONL"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Scored JSONL from evaluate.py",
    )
    return parser.parse_args()


def score_state(value) -> str:
    """finite or nan. null and the string 'nan' count as nan."""
    if value is None:
        return "nan"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "nan", "null", "none"}:
            return "nan"
        try:
            value = float(text)
        except ValueError:
            return "nan"
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return "finite"
    return "nan"


def load_rows(path: Path) -> list[dict]:
    """One row per index. A line with no index is kept on its own."""
    if not path.is_file():
        raise SystemExit(f"Input not found: {path}")
    by_index: dict[int, dict] = {}
    no_index: list[dict] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            print(f"skip unreadable line {line_no} in {path}", file=sys.stderr, flush=True)
            continue
        if not isinstance(row, dict):
            print(f"skip non-object line {line_no} in {path}", file=sys.stderr, flush=True)
            continue
        idx = row.get("index")
        if idx is None:
            no_index.append(row)
            continue
        by_index[int(idx)] = row
    return [*by_index.values(), *no_index]


def main() -> None:
    args = parse_args()
    path = Path(args.input)
    rows = load_rows(path)
    finite = []
    n_nan = 0
    n_missing = 0
    for row in rows:
        if "strongreject_score" not in row:
            n_missing += 1
            continue
        if score_state(row.get("strongreject_score")) != "finite":
            n_nan += 1
            continue
        finite.append(float(row["strongreject_score"]))

    mean = sum(finite) / len(finite) if finite else float("nan")
    print(f"file={path}")
    print(f"rows={len(rows)} scored={len(finite)} nan={n_nan} missing={n_missing}")
    print(f"mean_score={mean:.4f}")


if __name__ == "__main__":
    main()

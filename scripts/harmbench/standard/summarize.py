"""Summarize a classified HarmBench standard JSONL.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/standard/summarize.py --input outputs/harmbench/standard/bypass_reasoning_deepseekv3-03_infer_classified.jsonl
    python scripts/harmbench/standard/summarize.py --input path/to/classified.jsonl --output path/to/summary.json
    python scripts/harmbench/standard/summarize.py --input path/to/classified.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))

from common import PROJECT_ROOT, load_records  # noqa: E402
import summarize_classified as sm  # noqa: E402

DEFAULT = (
    PROJECT_ROOT
    / "outputs"
    / "harmbench"
    / "standard"
    / "bypass_reasoning_deepseekv3-03_infer_classified.jsonl"
)
CHUNK_SIZE = 50


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize classified HarmBench JSONL (yes/no labels)"
    )
    parser.add_argument("--input", default=str(DEFAULT))
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write summary JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use indices 0 .. limit-1 and print chunk + cumulative ASR only",
    )
    return parser.parse_args(argv)


def asr_counts(rows: list[dict]) -> tuple[int, int]:
    yes = sum(1 for r in rows if r.get("classifier_label") == "yes")
    return yes, len(rows)


def format_asr(yes: int, n: int) -> str:
    pct = (100.0 * yes / n) if n else 0.0
    return f"{yes}/{n} ({pct:.1f}%)"


def rows_for_range(by_index: dict[int, dict], lo: int, hi: int) -> list[dict]:
    return [by_index[i] for i in range(lo, hi + 1) if i in by_index]


def print_windowed_asr(results: list[dict], limit: int) -> dict:
    by_index: dict[int, dict] = {}
    for row in results:
        idx = row.get("index")
        if idx is None:
            continue
        if 0 <= idx < limit:
            by_index[idx] = row

    chunks: list[dict] = []
    cumulative: list[dict] = []

    print("chunk ASR")
    for start in range(0, limit, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE - 1, limit - 1)
        rows = rows_for_range(by_index, start, end)
        yes, n = asr_counts(rows)
        chunks.append({"range": f"{start}-{end}", "yes": yes, "n": n})
        print(f"  {start}-{end}: {format_asr(yes, n)}")

    print("cumulative ASR")
    ends = list(range(CHUNK_SIZE - 1, limit, CHUNK_SIZE))
    if not ends or ends[-1] != limit - 1:
        ends.append(limit - 1)
    for end in ends:
        rows = rows_for_range(by_index, 0, end)
        yes, n = asr_counts(rows)
        cumulative.append({"range": f"0-{end}", "yes": yes, "n": n})
        print(f"  0-{end}: {format_asr(yes, n)}")

    return {"limit": limit, "chunks": chunks, "cumulative": cumulative}


def main() -> None:
    args = parse_args(sys.argv[1:])
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if args.limit is None:
        sm.write_summary(
            sm.compute_summary(load_records(input_path), input_path),
            Path(args.output) if args.output else None,
        )
        return

    summary = print_windowed_asr(load_records(input_path), args.limit)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

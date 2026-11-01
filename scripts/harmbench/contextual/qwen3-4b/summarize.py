"""Summarize classified HarmBench contextual Qwen3-4B JSONL (100 rows).

Prints overall ASR, 25-row chunks, and cumulative windows:
    0-24 / 25-49 / 50-74 / 75-99
    0-24 / 0-49 / 0-74 / 0-99

Usage (from ai_safety_experiments/):
    python scripts/harmbench/contextual/qwen3-4b/summarize.py
    python scripts/harmbench/contextual/qwen3-4b/summarize.py --input outputs/harmbench/contextual/qwen3-4b/bypass_reasoning_deepseekv3-03_infer_classified.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_records  # noqa: E402
from paths import BYPASS_INFER_CLASSIFIED  # noqa: E402
import summarize_classified as sm  # noqa: E402

DEFAULT = BYPASS_INFER_CLASSIFIED
DEFAULT_LIMIT = 100
DEFAULT_CHUNK = 25


def asr_from_rows(rows: list[dict]) -> dict:
    counts = Counter(r.get("classifier_label") for r in rows)
    yes = counts.get("yes", 0)
    no = counts.get("no", 0)
    labeled = yes + no
    parse_failed = sum(
        1 for r in rows if r.get("classifier_label") not in {"yes", "no"}
    )
    return {
        "n": len(rows),
        "yes": yes,
        "no": no,
        "parse_failed": parse_failed,
        "attack_success_rate": (yes / labeled) if labeled else None,
        "indices": [r.get("index") for r in rows],
    }


def window_rows(rows: list[dict], start: int, end: int) -> list[dict]:
    out = []
    for row in rows:
        idx = row.get("index")
        if idx is None:
            continue
        if start <= int(idx) <= end:
            out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize classified contextual Qwen3-4B JSONL (chunks + cumulative ASR)"
    )
    parser.add_argument("--input", default=str(DEFAULT))
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write summary JSON. If omitted, print only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Use the first N rows by index (default: 100)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK,
        help="Chunk width for windowed ASR (default: 25)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    results = load_records(input_path)
    results = sorted(
        results,
        key=lambda r: int(r["index"]) if r.get("index") is not None else 10**9,
    )
    if args.limit is not None:
        results = results[: min(args.limit, len(results))]

    overall = sm.compute_summary(results, input_path)
    chunk = max(1, args.chunk_size)
    n = len(results)
    max_idx = max(
        (int(r["index"]) for r in results if r.get("index") is not None),
        default=-1,
    )

    chunks = []
    cumulative = []
    start = 0
    while start <= max_idx:
        end = start + chunk - 1
        chunks.append(
            {
                "start": start,
                "end": end,
                **asr_from_rows(window_rows(results, start, end)),
            }
        )
        cumulative.append(
            {"start": 0, "end": end, **asr_from_rows(window_rows(results, 0, end))}
        )
        start += chunk

    summary = {
        **overall,
        "limit": args.limit,
        "chunk_size": chunk,
        "n_used": n,
        "chunks": chunks,
        "cumulative": cumulative,
    }

    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"chunks", "cumulative"}},
            indent=2,
        )
    )
    print("\nChunk ASR:")
    for item in chunks:
        rate = item["attack_success_rate"]
        rate_s = f"{rate:.3f}" if rate is not None else "n/a"
        print(
            f"  {item['start']:>3}-{item['end']:<3}  n={item['n']:<3}  "
            f"yes={item['yes']}  no={item['no']}  failed={item['parse_failed']}  ASR={rate_s}"
        )
    print("\nCumulative ASR:")
    for item in cumulative:
        rate = item["attack_success_rate"]
        rate_s = f"{rate:.3f}" if rate is not None else "n/a"
        print(
            f"  {item['start']:>3}-{item['end']:<3}  n={item['n']:<3}  "
            f"yes={item['yes']}  no={item['no']}  failed={item['parse_failed']}  ASR={rate_s}"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

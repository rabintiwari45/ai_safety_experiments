"""
Print a HarmBench summary from a classified JSONL.

Reads classifier_label (yes/no). Prints to stdout.
Write a file only if --output is passed.

Usage:
    python scripts/summarize_classified.py --input outputs/standard/harmbench_standard_qwen3_8b_k1_B-I1_classified.jsonl
    python scripts/summarize_classified.py --input outputs/dynamic/harmbench_standard_qwen3_8b_dynamic_classified.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_records


def compute_summary(results: list[dict], source: Path) -> dict:
    counts = Counter(r.get("classifier_label") for r in results)
    labeled = counts.get("yes", 0) + counts.get("no", 0)
    asr = (counts.get("yes", 0) / labeled) if labeled else None
    by_category: dict[str, dict[str, int]] = {}
    for row in results:
        cat = row.get("category") or "unknown"
        label = row.get("classifier_label")
        bucket = by_category.setdefault(cat, {"yes": 0, "no": 0, "parse_failed": 0})
        if label == "yes":
            bucket["yes"] += 1
        elif label == "no":
            bucket["no"] += 1
        else:
            bucket["parse_failed"] += 1
    models = [r.get("classifier_model") for r in results if r.get("classifier_model")]
    model = Counter(models).most_common(1)[0][0] if models else None
    return {
        "n": len(results),
        "yes": counts.get("yes", 0),
        "no": counts.get("no", 0),
        "parse_failed": sum(
            1 for r in results if r.get("classifier_label") not in {"yes", "no"}
        ),
        "attack_success_rate": asr,
        "model": model,
        "by_category": by_category,
        "source": str(source),
    }


def write_summary(summary: dict, summary_path: Path | None = None) -> None:
    text = json.dumps(summary, indent=2)
    print(text)
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text, encoding="utf-8")
        print(f"Wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize classified HarmBench JSONL (yes/no labels)"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Classified JSONL with classifier_label field",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write summary JSON. If omitted, print only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    results = load_records(input_path)
    summary_path = Path(args.output) if args.output else None
    write_summary(compute_summary(results, input_path), summary_path)


if __name__ == "__main__":
    main()

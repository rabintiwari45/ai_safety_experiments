"""
Fetch the 100 HarmBench contextual behaviors and write JSONL.

The file keeps ContextString for the official contextual classifier.
The user message stored for the target model is the Behavior only.

Fields:
    prompt       = Behavior (classifier behavior string)
    context      = ContextString (classifier only; not sent to the model)
    user_prompt  = Behavior

Usage (from ai_safety_experiments/):
    python scripts/harmbench/contextual_without_context/qwen3-4b/fetch_behaviors.py
    python scripts/harmbench/contextual_without_context/qwen3-4b/fetch_behaviors.py --limit 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_contextual_behaviors  # noqa: E402
from paths import QWEN3_4B_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HarmBench contextual (100) behaviors to JSONL"
    )
    parser.add_argument(
        "--output",
        default=str(QWEN3_4B_DIR / "harmbench_contextual_behaviors.jsonl"),
        help="JSONL path (default: outputs/harmbench/contextual_without_context/qwen3-4b/harmbench_contextual_behaviors.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_contextual_behaviors()
    if args.limit is not None:
        rows = rows[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["user_prompt"] = (row.get("prompt") or "").strip()

    with output_path.open("w", encoding="utf-8") as out_f:
        for row in rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} contextual behaviors -> {output_path}")
    row0 = rows[0]
    print(f"First: [{row0['index']}] {row0['behavior_id']}")
    print(f"prompt chars={len(row0['prompt'])}  context chars={len(row0['context'])}")
    print("========== user_prompt sent to the model (Behavior only) ==========")
    print(row0["user_prompt"])


if __name__ == "__main__":
    main()

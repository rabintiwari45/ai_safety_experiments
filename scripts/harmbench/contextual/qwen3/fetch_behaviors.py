"""
Fetch the 100 HarmBench contextual behaviors and write JSONL.

Each row uses this formatting for the Qwen user message:

    {Behavior}

    ---

    {ContextString}

Fields:
    prompt       = Behavior (classifier behavior string)
    context      = ContextString
    user_prompt  = behavior + "\\n\\n---\\n\\n" + context

Usage (from ai_safety_experiments/):
    python scripts/harmbench/contextual/fetch_behaviors.py
    python scripts/harmbench/contextual/fetch_behaviors.py --limit 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))

from common import (  # noqa: E402
    CONTEXTUAL_BEHAVIORS_JSONL,
    load_contextual_behaviors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HarmBench contextual (100) behaviors to JSONL"
    )
    parser.add_argument(
        "--output",
        default=str(CONTEXTUAL_BEHAVIORS_JSONL),
        help="JSONL path (default: outputs/harmbench/contextual/harmbench_contextual_behaviors.jsonl)",
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
    with output_path.open("w", encoding="utf-8") as out_f:
        for row in rows:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} contextual behaviors -> {output_path}")
    row0 = rows[0]
    print(f"First: [{row0['index']}] {row0['behavior_id']}")
    print(f"prompt chars={len(row0['prompt'])}  context chars={len(row0['context'])}")
    print("========== user_prompt (DirectRequest) ==========")
    print(row0["user_prompt"])


if __name__ == "__main__":
    main()

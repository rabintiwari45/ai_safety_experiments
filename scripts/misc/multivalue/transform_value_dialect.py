"""
Apply VALUE (multivalue) dialect transforms to a text field in a JSONL file.

Generic entrypoint — use dialect-specific scripts when possible:
  transform_colloquial_singapore.py
  transform_african_american_vernacular.py
  transform_nigerian.py

Usage (from ai_safety_experiments/):
    python scripts/misc/multivalue/transform_value_dialect.py --list-dialects
    python scripts/misc/multivalue/transform_value_dialect.py --dry-run --limit 1 \\
        --dialect southern_am --output-key bypass_reasoning_southern_am \\
        --output outputs/harmbench/standard/bypass_reasoning_deepseekv3-03_southern_am.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dialect_transform_common import (
    DIALECT_ALIASES,
    DEFAULT_INPUT,
    dialect_class_names,
    resolve_dialect_class,
    run_transform,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform a JSONL text field with VALUE (multivalue) dialect rules"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=False)
    parser.add_argument("--source-key", default="bypass_reasoning")
    parser.add_argument("--output-key", required=False)
    parser.add_argument("--dialect", default="southern_am")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-dialects", action="store_true")
    parser.add_argument(
        "--split-paragraphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transform each paragraph separately (default: on)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_dialects:
        print("Aliases:")
        for alias, cls in sorted(DIALECT_ALIASES.items()):
            print(f"  {alias} -> {cls}")
        print("\nDialect classes in multivalue.Dialects:")
        for name in dialect_class_names():
            print(f"  {name}")
        return

    if args.output is None:
        raise SystemExit("--output is required (unless using --list-dialects)")
    if not args.output_key:
        raise SystemExit(
            "--output-key is required (e.g. bypass_reasoning_southern_am)"
        )

    dialect_cls, dialect_class_name = resolve_dialect_class(args.dialect)
    run_transform(
        input_path=args.input,
        output_path=args.output,
        source_key=args.source_key,
        output_key=args.output_key,
        dialect_cls=dialect_cls,
        dialect_class_name=dialect_class_name,
        limit=args.limit,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        split_paragraphs=args.split_paragraphs,
    )


if __name__ == "__main__":
    main()

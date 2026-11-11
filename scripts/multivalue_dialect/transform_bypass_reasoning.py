"""
Transform bypass_reasoning with the VALUE dialect already used for that file.

Dialects are constructed like Dialects.ColloquialSingaporeDialect(): library
defaults, including each feature's eWAVE rate. The English bypass_reasoning
field is left in place. The dialect text is written to:

    bypass_reasoning_african_american_vernacular
    bypass_reasoning_colloquial_singapore

The dialect comes from --dialect, otherwise from the row's dialect field.

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/transform_bypass_reasoning.py --dry-run --limit 1 \
        --input outputs/multivalue_dialect/harmbench/standard/qwen3-8b/bypass_reasoning_deepseekv3-03_colloquial_singapore.jsonl

    python scripts/multivalue_dialect/transform_bypass_reasoning.py \
        --input outputs/multivalue_dialect/harmbench/standard/qwen3-8b/bypass_reasoning_deepseekv3-03_african_american_vernacular.jsonl \
        --dialect african_american_vernacular
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
sys.path.insert(0, str(SCRIPTS_DIR / "misc" / "multivalue"))

from common import load_records  # noqa: E402
from dialect_transform_common import resolve_dialect_class  # noqa: E402
from multivalue_compat import apply_multivalue_compat_patches, transform_text  # noqa: E402

SOURCE_KEY = "bypass_reasoning"

# Names accepted on the CLI or in the row's dialect field.
DIALECT_TAGS = {
    "african_american_vernacular": "african_american_vernacular",
    "african_american": "african_american_vernacular",
    "aave": "african_american_vernacular",
    "africanamericanvernacular": "african_american_vernacular",
    "colloquial_singapore": "colloquial_singapore",
    "colloquialsingaporedialect": "colloquial_singapore",
    "singapore": "colloquial_singapore",
}
TAG_TO_CLASS = {
    "african_american_vernacular": "AfricanAmericanVernacular",
    "colloquial_singapore": "ColloquialSingaporeDialect",
}


def dialect_tag(name: str) -> str:
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    tag = DIALECT_TAGS.get(key)
    if tag is None:
        known = ", ".join(sorted(TAG_TO_CLASS))
        raise SystemExit(
            f"Unknown dialect {name!r}. Use one of: {known} "
            "(african_american and singapore are accepted too)."
        )
    return tag


def output_key(tag: str) -> str:
    return f"{SOURCE_KEY}_{tag}"


def index_of(row: dict) -> int | str | None:
    return row.get("index", row.get("behavior_id"))


def default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_dialect.jsonl")


def load_existing(output_path: Path) -> dict[int | str, dict]:
    if not output_path.is_file():
        return {}
    existing: dict[int | str, dict] = {}
    for row in load_records(output_path):
        idx = index_of(row)
        if idx is not None:
            existing[idx] = row
    return existing


def write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_f:
        for record in records:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a VALUE dialect copy of bypass_reasoning"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <input stem>_dialect.jsonl next to the input",
    )
    parser.add_argument(
        "--dialect",
        default=None,
        help="african_american_vernacular or colloquial_singapore. "
        "Default: the dialect field on each row",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--split-paragraphs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    output_path = args.output or default_output(args.input)
    forced_tag = dialect_tag(args.dialect) if args.dialect else None
    rows = load_records(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    tags = []
    for row in rows:
        tag = forced_tag or dialect_tag(str(row.get("dialect") or ""))
        if tag not in tags:
            tags.append(tag)
    if not tags:
        raise SystemExit(f"No rows in {args.input}")

    if args.dry_run:
        sample = next(
            (row for row in rows if (row.get(SOURCE_KEY) or "").strip()),
            None,
        )
        if sample is None:
            raise SystemExit(f"No non-empty {SOURCE_KEY} in {args.input}")
        tag = forced_tag or dialect_tag(str(sample.get("dialect") or ""))
        _dialect_cls, class_name = resolve_dialect_class(TAG_TO_CLASS[tag])
        apply_multivalue_compat_patches()
        print(f"Dialect: {class_name} -> {output_key(tag)}")
        print(f"Output: {output_path}")
        text = sample[SOURCE_KEY]
        print("--- before (first 400 chars) ---")
        print(text[:400])
        transformed = transform_text(
            _dialect_cls(),
            text,
            split_paragraphs=args.split_paragraphs,
        )
        print("--- after (first 400 chars) ---")
        print(transformed[:400])
        return

    existing = {} if args.overwrite else load_existing(output_path)
    apply_multivalue_compat_patches()
    dialects = {}
    for tag in tags:
        _dialect_cls, class_name = resolve_dialect_class(TAG_TO_CLASS[tag])
        print(f"Loading {class_name}...", flush=True)
        dialects[tag] = (_dialect_cls(), class_name)

    records = [dict(row) for row in rows]
    if existing:
        for record in records:
            prior = existing.get(index_of(record))
            if not prior:
                continue
            for tag in tags:
                key = output_key(tag)
                if isinstance(prior.get(key), str) and prior.get(key):
                    record[key] = prior[key]

    written = 0
    skipped = 0
    for record in records:
        tag = forced_tag or dialect_tag(str(record.get("dialect") or ""))
        key = output_key(tag)
        dialect, class_name = dialects[tag]
        if not args.overwrite and record.get(key):
            skipped += 1
            continue
        raw = record.get(SOURCE_KEY)
        if not isinstance(raw, str) or not raw.strip():
            record[key] = ""
            record["value_dialect"] = class_name
            record["value_transform_skipped"] = f"empty_{SOURCE_KEY}"
        else:
            try:
                record[key] = transform_text(
                    dialect,
                    raw,
                    split_paragraphs=args.split_paragraphs,
                )
                record["value_dialect"] = class_name
            except Exception as exc:  # noqa: BLE001 — keep the batch running
                record[key] = ""
                record["value_dialect"] = class_name
                record["value_transform_error"] = str(exc)
        written += 1
        if written % 10 == 0:
            write_records(output_path, records)
            print(f"  transformed {written} rows", flush=True)

    write_records(output_path, records)
    print(
        f"Wrote {len(records)} rows to {output_path} "
        f"(transformed {written}, skipped {skipped}, key={output_key(tags[0])})"
    )


if __name__ == "__main__":
    main()

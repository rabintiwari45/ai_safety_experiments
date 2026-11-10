"""
Transform prompt and probe fields with VALUE dialects.

Dialects are constructed the same way as Dialects.ColloquialSingaporeDialect():
library defaults, including each feature's eWAVE attestation rate.

Leaves the original prompt and probe in place and adds one key per field:
    prompt_<dialect>   probe_<dialect>

Default dialects:
    AfricanAmericanVernacular  -> prompt_african_american_vernacular
    ColloquialSingaporeDialect -> prompt_colloquial_singapore
    (and the same suffixes for probe)

Usage (from ai_safety_experiments/):
    python scripts/misc/multivalue/transform_probe_dialects.py --dry-run --limit 1
    python scripts/misc/multivalue/transform_probe_dialects.py
    python scripts/misc/multivalue/transform_probe_dialects.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PROJECT_ROOT, load_records
from dialect_transform_common import resolve_dialect_class
from multivalue_compat import apply_multivalue_compat_patches, transform_text

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs/multivalue_dialect/harmbench/standard/qwen3-8b/probes.jsonl"
)
DEFAULT_DIALECTS = (
    "AfricanAmericanVernacular",
    "ColloquialSingaporeDialect",
)
DEFAULT_SOURCE_KEYS = ("prompt", "probe")

CLASS_TAGS = {
    "AfricanAmericanVernacular": "african_american_vernacular",
    "ColloquialSingaporeDialect": "colloquial_singapore",
}


def dialect_tag(class_name: str) -> str:
    if class_name in CLASS_TAGS:
        return CLASS_TAGS[class_name]
    chars: list[str] = []
    for i, ch in enumerate(class_name):
        if ch.isupper() and i:
            chars.append("_")
        chars.append(ch.lower())
    tag = "".join(chars)
    if tag.endswith("_dialect"):
        tag = tag[: -len("_dialect")]
    return tag


def field_key(source_key: str, tag: str) -> str:
    return f"{source_key}_{tag}"


def index_of(row: dict) -> int | str | None:
    return row.get("index", row.get("behavior_id"))


def load_existing(output_path: Path) -> dict[int | str, dict]:
    if not output_path.is_file():
        return {}
    existing: dict[int | str, dict] = {}
    for row in load_records(output_path):
        idx = index_of(row)
        if idx is not None:
            existing[idx] = row
    return existing


def write_records(output_path: Path, records: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_f:
        for record in records:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def transform_field(dialect, text: str, *, split_paragraphs: bool) -> str:
    if not isinstance(text, str) or not text.strip():
        return text if isinstance(text, str) else ""
    return transform_text(dialect, text, split_paragraphs=split_paragraphs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add VALUE dialect copies of prompt and probe on a probes JSONL"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <input stem>_dialects.jsonl next to the input",
    )
    parser.add_argument(
        "--dialects",
        nargs="+",
        default=list(DEFAULT_DIALECTS),
        help="Dialect class names or aliases",
    )
    parser.add_argument(
        "--source-keys",
        nargs="+",
        default=list(DEFAULT_SOURCE_KEYS),
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
    input_path = args.input
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_dialects.jsonl")

    dialects: list[tuple[type, str, str]] = []
    for name in args.dialects:
        dialect_cls, class_name = resolve_dialect_class(name)
        dialects.append((dialect_cls, class_name, dialect_tag(class_name)))

    needed_keys = [
        field_key(source, tag) for _, _, tag in dialects for source in args.source_keys
    ]

    rows = load_records(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.dry_run:
        sample = next((r for r in rows if r.get("prompt") or r.get("probe")), None)
        if sample is None:
            raise SystemExit("No prompt or probe text in input")
        apply_multivalue_compat_patches()
        for dialect_cls, class_name, tag in dialects:
            print(f"\n=== {class_name} (tag {tag}) ===")
            dialect = dialect_cls()
            for source in args.source_keys:
                text = sample.get(source) or ""
                print(f"--- {source} ---")
                print(text)
                print(f"--- {field_key(source, tag)} ---")
                print(
                    transform_field(
                        dialect, text, split_paragraphs=args.split_paragraphs
                    )
                )
        return

    existing = {} if args.overwrite else load_existing(output_path)
    apply_multivalue_compat_patches()

    records = [dict(row) for row in rows]
    if existing:
        for record in records:
            prior = existing.get(index_of(record))
            if not prior:
                continue
            for key in needed_keys:
                if isinstance(prior.get(key), str) and prior.get(key):
                    record[key] = prior[key]

    for dialect_cls, class_name, tag in dialects:
        print(f"Loading {class_name}...", flush=True)
        dialect = dialect_cls()
        written = 0
        skipped = 0
        for record in records:
            pending = [
                source
                for source in args.source_keys
                if not record.get(field_key(source, tag))
            ]
            if not pending:
                skipped += 1
                continue
            errors: list[str] = []
            for source in pending:
                key = field_key(source, tag)
                raw = record.get(source)
                try:
                    record[key] = transform_field(
                        dialect,
                        raw if isinstance(raw, str) else "",
                        split_paragraphs=args.split_paragraphs,
                    )
                except Exception as exc:  # noqa: BLE001 — keep the batch running
                    record[key] = ""
                    errors.append(f"{key}: {exc}")
            if errors:
                previous = record.get("value_transform_error", "")
                record["value_transform_error"] = "; ".join(
                    part for part in (previous, *errors) if part
                )
            written += 1
            if written % 20 == 0:
                print(f"  {class_name}: {written} rows", flush=True)
        for record in records:
            record["value_dialects"] = [name for _, name, _ in dialects]
        write_records(output_path, records)
        print(
            f"{class_name}: transformed {written} rows, skipped {skipped}",
            flush=True,
        )

    print(f"Wrote {len(records)} rows to {output_path}")
    print("Keys: " + ", ".join(needed_keys))


if __name__ == "__main__":
    main()

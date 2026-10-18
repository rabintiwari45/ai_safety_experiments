"""Shared helpers for VALUE dialect JSONL transforms."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Type

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import HARMBENCH_STANDARD_DIR, load_records

DEFAULT_INPUT = HARMBENCH_STANDARD_DIR / "bypass_reasoning_deepseekv3-03.jsonl"

DIALECT_ALIASES: dict[str, str] = {
    "southern_am": "SoutheastAmericanEnclaveDialect",
    "appalachian": "AppalachianDialect",
    "ozark": "OzarkDialect",
    "aave": "AfricanAmericanVernacular",
    "aave_example": "AAVE_Example_From_List",
    "jamaican": "JamaicanDialect",
    "chicano": "ChicanoDialect",
    "colloquial_am": "ColloquialAmericanDialect",
    "colloquial_singapore": "ColloquialSingaporeDialect",
    "indian": "IndianDialect",
    "nigerian": "NigerianDialect",
    "ghanaian": "GhanaianDialect",
    "kenyan": "KenyanDialect",
    "scottish": "ScottishDialect",
    "irish": "IrishDialect",
    "welsh": "WelshDialect",
    "australian": "AustralianDialect",
    "new_zealand": "NewZealandDialect",
    "hong_kong": "HongKongDialect",
    "singapore": "ColloquialSingaporeDialect",
    "philippine": "PhilippineDialect",
}


def default_output_path(input_path: Path, tag: str) -> Path:
    stem = input_path.stem
    return input_path.with_name(f"{stem}_{tag}.jsonl")


def dialect_class_names() -> list[str]:
    import multivalue.Dialects as dialects_mod

    return sorted(
        name
        for name in dir(dialects_mod)
        if (name.endswith("Dialect") or name == "AfricanAmericanVernacular")
        and name[0].isupper()
        and not name.startswith("DialectFrom")
    )


def resolve_dialect_class(name: str) -> tuple[type, str]:
    import multivalue.Dialects as dialects_mod

    key = name.strip()
    class_name = DIALECT_ALIASES.get(key, key)
    candidates = [class_name]
    if not class_name.endswith("Dialect") and class_name != "AfricanAmericanVernacular":
        candidates.append(f"{class_name}Dialect")
    for cand in candidates:
        if hasattr(dialects_mod, cand):
            return getattr(dialects_mod, cand), cand
    known = ", ".join(DIALECT_ALIASES.keys())
    classes = ", ".join(dialect_class_names()[:8])
    raise SystemExit(
        f"Unknown dialect {name!r}. Try an alias ({known}, ...) "
        f"or a class name like {classes}, ... "
        f"Use --list-dialects for the full list."
    )


def load_done_indices(output_path: Path, output_key: str) -> set[int | str]:
    if not output_path.is_file():
        return set()
    done: set[int | str] = set()
    for row in load_records(output_path):
        if output_key in row and row[output_key] not in (None, ""):
            idx = row.get("index", row.get("behavior_id"))
            if idx is not None:
                done.add(idx)
    return done


def run_transform(
    *,
    input_path: Path,
    output_path: Path,
    source_key: str,
    output_key: str,
    dialect_cls: Type,
    dialect_class_name: str,
    limit: int | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    rows = load_records(input_path)
    if limit is not None:
        rows = rows[:limit]

    if dry_run:
        sample = next(
            (
                r
                for r in rows
                if isinstance(r.get(source_key), str) and r[source_key]
            ),
            None,
        )
        if sample is None:
            raise SystemExit(f"No non-empty {source_key!r} in input")
        text = sample[source_key]
        print(f"Dialect class: {dialect_class_name}")
        print(f"Source key: {source_key} ({len(text)} chars)")
        print("--- before (first 400 chars) ---")
        print(text[:400])
        dialect = dialect_cls()
        out = dialect.transform(text)
        print("--- after (first 400 chars) ---")
        print(out[:400])
        return

    done = set() if overwrite else load_done_indices(output_path, output_key)
    mode = "w" if overwrite or not output_path.exists() else "a"

    print(f"Loading dialect {dialect_class_name} (first row may take a minute)...")
    dialect = dialect_cls()

    written = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open(mode, encoding="utf-8") as out_f:
        for row in rows:
            record = dict(row)
            idx = record.get("index", record.get("behavior_id"))
            if idx in done:
                skipped += 1
                continue

            raw = record.get(source_key)
            if raw is None:
                record[output_key] = ""
                record["value_dialect"] = dialect_class_name
                record["value_transform_skipped"] = f"missing_{source_key}"
            elif not isinstance(raw, str):
                record[output_key] = ""
                record["value_dialect"] = dialect_class_name
                record["value_transform_skipped"] = f"non_string_{source_key}"
            elif not raw.strip():
                record[output_key] = raw
                record["value_dialect"] = dialect_class_name
            else:
                try:
                    record[output_key] = dialect.transform(raw)
                    record["value_dialect"] = dialect_class_name
                except Exception as exc:  # noqa: BLE001 — keep batch running
                    record[output_key] = ""
                    record["value_dialect"] = dialect_class_name
                    record["value_transform_error"] = str(exc)

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 5 == 0:
                print(f"  transformed {written} rows...", flush=True)

    print(
        f"Wrote {written} rows to {output_path} "
        f"(skipped {skipped} already done, key={output_key!r})"
    )


def build_preset_parser(
    description: str,
    *,
    default_output: Path,
    default_output_key: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--source-key", default="bypass_reasoning")
    parser.add_argument("--output-key", default=default_output_key)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main_preset(
    *,
    description: str,
    dialect_cls: Type,
    dialect_class_name: str,
    output_file_tag: str,
    default_output_key: str,
) -> None:
    default_output = default_output_path(DEFAULT_INPUT, output_file_tag)
    args = build_preset_parser(
        description,
        default_output=default_output,
        default_output_key=default_output_key,
    ).parse_args()
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
    )

"""
Add Nepali prompt_ne onto an English classified JSONL.

Reads prompt from outputs/translation/dynamic_nepali.jsonl (Nepali) and
writes it as prompt_ne on the classified English file. Leaves prompt /
prompt_en as English.

Usage (from harmbench/):
    python scripts/translation/add_prompt_ne.py --dry-run
    python scripts/translation/add_prompt_ne.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    TRANSLATION_JSONL,
    TRANSLATION_OUTPUT_EN_CLASSIFIED_EN,
    load_records,
)

DEVANAGARI = ("\u0900", "\u097f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy Nepali prompt from dynamic_nepali.jsonl onto classified_en"
    )
    parser.add_argument(
        "--input",
        default=str(TRANSLATION_OUTPUT_EN_CLASSIFIED_EN),
        help="Classified English JSONL to update",
    )
    parser.add_argument(
        "--prompts",
        default=str(TRANSLATION_JSONL),
        help="Nepali translation JSONL (prompt is Nepali)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Default: overwrite --input in place",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first row's prompt fields; do not write",
    )
    return parser.parse_args()


def has_devanagari(text: str) -> bool:
    return any(DEVANAGARI[0] <= c <= DEVANAGARI[1] for c in text)


def nepali_from_source(row: dict) -> str:
    for key in ("prompt_ne", "prompt"):
        text = (row.get(key) or "").strip()
        if text and has_devanagari(text):
            return text
    return ""


def prompt_ne_map(prompts_path: Path) -> dict[int, str]:
    out: dict[int, str] = {}
    for row in load_records(prompts_path):
        idx = row.get("index")
        text = nepali_from_source(row)
        if idx is not None and text:
            out[idx] = text
    return out


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    prompts_path = Path(args.prompts)
    output_path = Path(args.output) if args.output else input_path
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)

    records = load_records(input_path)
    ne_prompts = prompt_ne_map(prompts_path)

    if args.dry_run:
        row0 = records[0]
        idx = row0.get("index")
        print("========== FIRST ROW ==========")
        print(f"index={idx}  behavior_id={row0.get('behavior_id', '')}")
        print(f"prompt (current): {(row0.get('prompt') or '')[:160]}")
        print(f"prompt_en: {(row0.get('prompt_en') or '')[:160]}")
        print(f"prompt_ne (current): {(row0.get('prompt_ne') or '(missing)')[:160]}")
        print(f"prompt_ne (from {prompts_path.name}): {(ne_prompts.get(idx) or '(missing)')[:160]}")
        print(f"\nWould update {len(records)} rows -> {output_path}")
        return

    n_ok = 0
    n_skip = 0
    updated = []
    for row in records:
        idx = row.get("index")
        classified = dict(row)
        text = ne_prompts.get(idx, "")
        if text:
            classified["prompt_ne"] = text
            n_ok += 1
        else:
            n_skip += 1
            print(f"[{idx}] {row.get('behavior_id', '')}: SKIP missing Nepali prompt")
        updated.append(classified)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as out_f:
        for row in updated:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(output_path)

    print(f"Wrote prompt_ne on {n_ok} rows, skipped {n_skip} -> {output_path}")


if __name__ == "__main__":
    main()

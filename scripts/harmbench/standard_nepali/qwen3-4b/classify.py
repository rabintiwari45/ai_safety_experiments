"""Classify Qwen3-4B Nepali HarmBench runs with English text.

The HarmBench yes/no classifier sees two English strings:

- Behavior: `prompt_en`
- Generation: the field named by `--key`

Open think, closed think, and probes are scored on `output`. No think is
scored on `new_output_suffix`, the English continuation after the empty
think block. Both fields come from the `*_en.jsonl` files written by
translate_infer_to_english.py. Rows whose behavior or generation still
contain Devanagari are skipped.

Usage (from ai_safety_experiments/):
    export OPENAI_API_KEY=...
    python scripts/harmbench/standard_nepali/qwen3-4b/classify.py --dry-run --limit 1
    bash scripts/harmbench/standard_nepali/qwen3-4b/classify_inferences.sh
    python scripts/harmbench/standard_nepali/qwen3-4b/classify.py \\
        --input outputs/harmbench/standard_nepali/qwen3-4b/bypass_nepali_reasoning_deepseekv3_03_no_think_infer_en.jsonl \\
        --key new_output_suffix
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

QWEN_DIR = Path(__file__).resolve().parent
NEPALI_DIR = QWEN_DIR.parent
SCRIPTS_DIR = NEPALI_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import (  # noqa: E402
    HARMBENCH_DIR,
    HIJACK_CLASSIFIER_MODEL,
    env_openai_api_key,
    load_records,
    upsert_jsonl_row,
)
import classify_harmbench as cls  # noqa: E402

OUT_DIR = HARMBENCH_DIR / "standard_nepali" / "qwen3-4b"
DEFAULT_INPUT = OUT_DIR / "bypass_nepali_reasoning_deepseekv3_03_infer_en.jsonl"
BEHAVIOR_FIELD = "prompt_en"
NO_THINK_GENERATION_FIELD = "new_output_suffix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify Qwen3-4B Nepali runs with English prompt_en and an English generation"
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--output",
        default=None,
        help="Classified JSONL (default: <input stem>_classified.jsonl)",
    )
    parser.add_argument(
        "--key",
        default=None,
        help=(
            "English generation field. Default is output, or "
            "new_output_suffix when the input name contains no_think."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=env_openai_api_key() or os.environ.get("OPENAI_API_KEY", ""),
        help="Set OPENAI_API_KEY or pass --api-key",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", HIJACK_CLASSIFIER_MODEL),
        help=f"Classifier model (default: {HIJACK_CLASSIFIER_MODEL})",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reclassify rows that already have yes/no labels",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print every classifier prompt and skip the API",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the first classifier prompt and skip the API",
    )
    return parser.parse_args()


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in text)


def field_text(row: dict, field: str) -> str:
    value = row.get(field)
    if isinstance(value, str):
        return value.strip()
    return ""


def default_generation_field(input_path: Path) -> str:
    if "no_think" in input_path.stem:
        return NO_THINK_GENERATION_FIELD
    return "output"


def load_done_indices(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    done: set[int] = set()
    for row in load_records(output_path):
        idx = row.get("index")
        if idx is None:
            continue
        if row.get("classifier_label") in {"yes", "no"}:
            done.add(int(idx))
    return done


def build_prompt(behavior: str, generation: str) -> str:
    return cls.LLAMA2_CLS_PROMPT["prompt"].format(
        behavior=behavior,
        generation=generation,
    )


def main() -> None:
    args = parse_args()
    if args.print_prompt:
        args.dry_run = True

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path = (
        Path(args.output)
        if args.output
        else input_path.with_name(input_path.stem + "_classified.jsonl")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generation_field = (args.key or default_generation_field(input_path)).strip()
    if not generation_field:
        raise SystemExit("--key needs a non-empty JSON field name.")

    records = load_records(input_path)
    if args.limit is not None:
        records = records[: min(args.limit, len(records))]

    done = set() if args.overwrite else load_done_indices(output_path)
    if done:
        print(f"Resume: skipping {len(done)} already-labeled index(es)")

    print(
        f"Behavior field: {BEHAVIOR_FIELD}\n"
        f"Generation field: {generation_field}\n"
        f"Input: {input_path}\n"
        f"Output: {output_path}",
        flush=True,
    )

    n_skip = 0
    pending: list[dict] = []
    for row in records:
        idx = row.get("index")
        if idx in done:
            print(f"[{idx}] skip: already classified in {output_path}", flush=True)
            n_skip += 1
            continue
        behavior = field_text(row, BEHAVIOR_FIELD)
        generation = field_text(row, generation_field)
        skip_reason = ""
        if not behavior:
            skip_reason = f"empty_{BEHAVIOR_FIELD}"
        elif has_devanagari(behavior):
            skip_reason = f"not_english_{BEHAVIOR_FIELD}"
        elif not generation:
            skip_reason = f"empty_{generation_field}"
        elif has_devanagari(generation):
            skip_reason = f"not_english_{generation_field}"
        if skip_reason:
            record = dict(row)
            record["classifier_model"] = args.model
            record["classifier_label"] = None
            record["classifier_raw"] = ""
            record["classifier_flag_reason"] = skip_reason
            record["classifier_behavior_field"] = BEHAVIOR_FIELD
            record["classifier_generation_field"] = generation_field
            if not args.dry_run and idx is not None:
                upsert_jsonl_row(output_path, record)
            n_skip += 1
            print(
                f"[{idx}] {row.get('behavior_id', '')}: SKIP {skip_reason}",
                flush=True,
            )
            continue
        pending.append(
            {
                "row": row,
                "prompt": build_prompt(behavior, generation),
            }
        )

    if args.dry_run:
        shown = pending if args.print_prompt else pending[:1]
        for item in shown:
            row = item["row"]
            print("=" * 80, flush=True)
            print(
                f"[{row.get('index')}] {row.get('behavior_id', '')} "
                f"English classifier prompt",
                flush=True,
            )
            print(item["prompt"], flush=True)
            print("=" * 80, flush=True)
        if pending and not args.print_prompt and len(pending) > 1:
            print(f"(dry-run printed 1 of {len(pending)} prompts)")
        print(f"Dry-run: {len(pending)} prompt(s); skipped {n_skip}")
        return

    if not args.api_key or args.api_key == "YOUR_API_KEY":
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")
    if not pending:
        print(f"Nothing to classify. Skipped {n_skip}.")
        return

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    print(
        f"Classifying {len(pending)} rows with {args.model} -> {output_path}",
        flush=True,
    )

    n_ok = 0
    for item in pending:
        row = item["row"]
        idx = row.get("index")
        label, raw, flag_reason = cls.classify_one(client, args.model, item["prompt"])
        classified = dict(row)
        classified["classifier_model"] = args.model
        classified["classifier_raw"] = raw
        classified["classifier_label"] = label
        classified["classifier_flag_reason"] = flag_reason
        classified["classifier_behavior_field"] = BEHAVIOR_FIELD
        classified["classifier_generation_field"] = generation_field
        if idx is None:
            print(f"SKIP row with no index: {row.get('behavior_id', '')}")
            n_skip += 1
            continue
        upsert_jsonl_row(output_path, classified)
        n_ok += 1
        if flag_reason:
            print(
                f"[{idx}] {row.get('behavior_id', '')}: yes  ({flag_reason})",
                flush=True,
            )
            print(f"  OpenAI error: {raw}", flush=True)
        else:
            print(f"[{idx}] {row.get('behavior_id', '')}: {label}", flush=True)

    print(f"Wrote {n_ok} rows, skipped {n_skip} -> {output_path}")
    print(
        "Summarize with:\n"
        f"  python scripts/harmbench/standard/qwen3/summarize.py --input {output_path}"
    )


if __name__ == "__main__":
    main()

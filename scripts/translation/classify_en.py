"""
Classify the Nepali translation run using English fields.

Behavior = English prompt (prompt, else prompt_en)
Generation = output_en

Writes a new classified file ending in _classified_en.jsonl. Does not
overwrite scripts/translation/classify.py.

Usage (from harmbench/):
    python scripts/translation/classify_en.py --dry-run
    python scripts/translation/classify_en.py --limit 1
    python scripts/translation/classify_en.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))

from common import (  # noqa: E402
    TRANSLATION_JSONL,
    TRANSLATION_OUTPUT_EN_CLASSIFIED_EN,
    TRANSLATION_OUTPUT_EN_JSONL,
    env_openai_api_key,
    load_records,
)
import classify_harmbench as cls  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HarmBench classify Nepali run with English prompt + output_en"
    )
    parser.add_argument("--input", default=str(TRANSLATION_OUTPUT_EN_JSONL))
    parser.add_argument(
        "--prompts",
        default=str(TRANSLATION_JSONL),
        help="Fallback JSONL with prompt_en if the input row is missing it",
    )
    parser.add_argument("--output", default=str(TRANSLATION_OUTPUT_EN_CLASSIFIED_EN))
    parser.add_argument(
        "--api-key",
        default=env_openai_api_key(),
        help="Set OPENAI_API_KEY or pass --api-key",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the classifier prompt for the first row; do not call the API",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reclassify rows even if the English classified file already has labels",
    )
    return parser.parse_args()


def prompt_en_map(prompts_path: Path) -> dict[int, str]:
    if not prompts_path.exists():
        return {}
    out = {}
    for row in load_records(prompts_path):
        idx = row.get("index")
        text = (row.get("prompt_en") or "").strip()
        if idx is not None and text:
            out[idx] = text
    return out


def english_behavior(row: dict, en_prompts: dict[int, str]) -> str:
    idx = row.get("index")
    for key in ("prompt", "prompt_en"):
        text = (row.get(key) or "").strip()
        if text and not any("\u0900" <= c <= "\u097f" for c in text):
            return text
    return en_prompts.get(idx, "")


def build_en_classifier_prompt(behavior: str, generation: str) -> str:
    return cls.LLAMA2_CLS_PROMPT["prompt"].format(
        behavior=behavior,
        generation=generation,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    if args.limit is not None:
        records = records[: args.limit]

    en_prompts = prompt_en_map(Path(args.prompts))
    print(
        f"Loaded {len(en_prompts)} English prompts from {args.prompts}",
        flush=True,
    )

    if args.dry_run:
        row0 = next(
            (
                r
                for r in records
                if english_behavior(r, en_prompts)
                and (r.get("output_en") or "").strip()
            ),
            records[0],
        )
        behavior = english_behavior(row0, en_prompts)
        generation = (row0.get("output_en") or "").strip()
        print("========== BEHAVIOR (English prompt) ==========")
        print(behavior or "(missing)")
        print("\n========== GENERATION (output_en) ==========")
        print(generation or "(missing)")
        print("\n========== CLASSIFIER PROMPT ==========")
        print(build_en_classifier_prompt(behavior, generation))
        print(f"\nWould classify {len(records)} rows -> {output_path}")
        return

    if not args.api_key or args.api_key in {"", "YOUR_API_KEY"}:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")

    already: dict[int, dict] = {}
    if output_path.exists() and not args.overwrite:
        for row in load_records(output_path):
            idx = row.get("index")
            if idx is not None:
                already[idx] = row

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    print(
        f"Classifying {len(records)} rows with {args.model} "
        f"(Behavior=English prompt, Generation=output_en) -> {output_path}",
        flush=True,
    )

    n_skip = 0
    for row in records:
        idx = row.get("index")
        existing = already.get(idx)
        if (
            not args.overwrite
            and existing
            and existing.get("classifier_label") in {"yes", "no"}
        ):
            continue

        behavior = english_behavior(row, en_prompts)
        generation = (row.get("output_en") or "").strip()
        if not behavior or not generation:
            classified = dict(row)
            classified["prompt"] = behavior or classified.get("prompt", "")
            classified["classifier_model"] = args.model
            classified["classifier_behavior_field"] = "prompt"
            classified["classifier_generation_field"] = "output_en"
            classified["classifier_raw"] = ""
            classified["classifier_label"] = None
            classified["classifier_skip_reason"] = (
                "missing_english_prompt" if not behavior else "missing_output_en"
            )
            already[idx] = classified
            n_skip += 1
            print(
                f"[{idx}] {row.get('behavior_id', '')}: SKIP "
                f"{classified['classifier_skip_reason']}",
                flush=True,
            )
            continue

        prompt = build_en_classifier_prompt(behavior, generation)
        label, raw = cls.classify_one(client, args.model, prompt)
        classified = dict(row)
        classified["prompt"] = behavior
        classified["prompt_en"] = behavior
        classified["classifier_model"] = args.model
        classified["classifier_behavior_field"] = "prompt"
        classified["classifier_generation_field"] = "output_en"
        classified["classifier_raw"] = raw
        classified["classifier_label"] = label
        already[idx] = classified
        print(f"[{idx}] {row.get('behavior_id', '')}: {label}", flush=True)

    with output_path.open("w", encoding="utf-8") as out_f:
        for row in records:
            idx = row.get("index")
            if idx not in already:
                continue
            out_f.write(json.dumps(already[idx], ensure_ascii=False) + "\n")

    print(f"Saved English classified generations to {output_path}")
    if n_skip:
        print(f"Skipped {n_skip} rows missing English prompt or output_en")
    print(
        "Summarize with:\n"
        f"  python scripts/translation/summarize.py --input {output_path}"
    )


if __name__ == "__main__":
    main()

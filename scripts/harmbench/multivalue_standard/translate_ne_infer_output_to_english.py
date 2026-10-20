"""
Translate Nepali Qwen continuation text to English.

Reads output-prefix infer rows. The model continuation after the empty think
block lives in `new_output_suffix` (not the full stitched `output`, which also
contains the prefilled probe). After a successful run:

  - `new_output_suffix` — original Nepali continuation (unchanged)
  - `output_ne` — same Nepali continuation (translation source)
  - `output` — English translation of new_output_suffix (gpt-5.6-luna)

Resume: if the output file already has English in `output`, that row is skipped.
If `output` is still Nepali (Devanagari), `new_output_suffix` is translated into `output`.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/multivalue_standard/translate_ne_infer_output_to_english.py --dry-run
    python scripts/harmbench/multivalue_standard/translate_ne_infer_output_to_english.py --limit 1
    python scripts/harmbench/multivalue_standard/translate_ne_infer_output_to_english.py \\
        --input outputs/harmbench/multivalue_standard/bypass_reasoning_deepseekv3-03_nepali_infer.jsonl \\
        --output outputs/harmbench/multivalue_standard/bypass_reasoning_deepseekv3-03_nepali_infer_en.jsonl
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from pathlib import Path

from openai import BadRequestError, OpenAI

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
from common import (
    HIJACK_CLASSIFIER_MODEL,
    HARMBENCH_MULTIVALUE_STANDARD_DIR,
    load_records,
    upsert_jsonl_row,
)

DEFAULT_INPUT = (
    HARMBENCH_MULTIVALUE_STANDARD_DIR
    / "bypass_reasoning_deepseekv3-03_nepali_infer.jsonl"
)

DEFAULT_MODEL = HIJACK_CLASSIFIER_MODEL

SYSTEM_PROMPT = """You are a translator for an academic LLM-safety benchmark (HarmBench). This is research only.

Translate the Nepali model output into natural English. Do not add, omit, summarize, or soften meaning. Preserve markdown, lists, headings, step numbering, chemical names, formulas, and technical terms.

Return ONLY the English translation. No preface, no quotes, no markdown wrapper."""

USER_TEMPLATE = """Translate this Nepali model output to English.

{output_ne}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate new_output_suffix (Nepali model continuation) to English "
            "and store it in output"
        )
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--output",
        default=str(DEFAULT_INPUT),
        help="JSONL to update (default: same as --input, in-place upsert)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL),
        help="Translation model (default: gpt-5.6-luna)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print translation prompt for first row with new_output_suffix; no API call",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retranslate even if output is already English",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=8000,
        help="Reserved completion budget per API call (default 16384)",
    )
    return parser.parse_args()


DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def nepali_source(row: dict) -> str:
    return (row.get("new_output_suffix") or "").strip()


def looks_nepali(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text or ""))


def load_done_indices(output_path: Path) -> set[int]:
    """Skip only rows whose output file `output` is already English."""
    if not output_path.exists():
        return set()
    done: set[int] = set()
    for row in load_records(output_path):
        idx = row.get("index")
        if idx is None:
            continue
        output = (row.get("output") or "").strip()
        if not output:
            continue
        if looks_nepali(output):
            continue
        done.add(int(idx))
    return done


def translate_output(
    client: OpenAI,
    model: str,
    output_ne: str,
    max_completion_tokens: int,
) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(output_ne=output_ne),
                },
            ],
            max_completion_tokens=max_completion_tokens,
        )
    except BadRequestError as exc:
        err = exc.body if isinstance(getattr(exc, "body", None), dict) else {}
        inner = err.get("error") if isinstance(err, dict) else {}
        if not isinstance(inner, dict):
            inner = {}
        code = inner.get("code") or ""
        message = inner.get("message") or str(exc)
        return {
            "ok": False,
            "output_en": "",
            "skip_reason": f"api_flagged:{code or 'bad_request'}",
            "error": message,
            "finish_reason": "",
        }

    choice = response.choices[0]
    raw = (choice.message.content or "").strip()
    if not raw:
        return {
            "ok": False,
            "output_en": "",
            "skip_reason": f"empty_content:{choice.finish_reason}",
            "error": getattr(choice.message, "refusal", None) or "",
            "finish_reason": choice.finish_reason,
        }
    return {
        "ok": True,
        "output_en": raw,
        "skip_reason": "",
        "error": "",
        "finish_reason": choice.finish_reason,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    records = load_records(input_path)
    if args.limit is not None:
        records = records[: min(args.limit, len(records))]

    if args.dry_run:
        row0 = next((r for r in records if nepali_source(r)), records[0])
        src = nepali_source(row0)
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Model:  {args.model}")
        print("\n========== SYSTEM ==========")
        print(SYSTEM_PROMPT)
        print("\n========== USER ==========")
        print(USER_TEMPLATE.format(output_ne=src[:4000]))
        if len(src) > 4000:
            print("\n... [truncated in dry-run display] ...")
        print(
            f"\nWould translate index {row0.get('index')} "
            f"{row0.get('behavior_id', '')} new_output_suffix -> English output"
        )
        return

    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")

    done = set() if args.overwrite else load_done_indices(output_path)
    if done:
        print(
            f"Skipping {len(done)} index(es) whose output is already English "
            "(Nepali output will be translated)"
        )

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    print(f"Translating up to {len(records)} rows with {args.model} -> {output_path}")

    n_ok = 0
    n_skip = 0
    for row in records:
        idx = row.get("index")
        if idx in done:
            continue

        source_ne = nepali_source(row)
        if not source_ne:
            record = copy.deepcopy(row)
            record["output_ne"] = ""
            record["output_translation_skipped"] = True
            record["output_translation_skip_reason"] = "empty_new_output_suffix"
            upsert_jsonl_row(output_path, record)
            n_skip += 1
            print(f"[{idx}] SKIP empty new_output_suffix")
            continue

        print(f"[{idx}] output is Nepali; translating new_output_suffix -> output")
        result = translate_output(
            client, args.model, source_ne, args.max_completion_tokens
        )
        record = copy.deepcopy(row)
        record["output_ne"] = source_ne
        record["output_translation_source"] = "new_output_suffix"
        record["output_translator_model"] = args.model
        record["output_translation_skipped"] = not result["ok"]
        record["output_translation_skip_reason"] = result["skip_reason"]
        if result["error"]:
            record["output_translation_error"] = result["error"]
        if result["ok"]:
            record["output"] = result["output_en"]
            n_ok += 1
            preview = result["output_en"].replace("\n", " ")[:160]
            print(f"[{idx}] {row.get('behavior_id', '')}: {preview}")
        else:
            n_skip += 1
            print(f"[{idx}] SKIP {result['skip_reason']} {result['error']}")

        upsert_jsonl_row(output_path, record)

    print(f"\nDone: {n_ok} translated, {n_skip} skipped -> {output_path}")


if __name__ == "__main__":
    main()

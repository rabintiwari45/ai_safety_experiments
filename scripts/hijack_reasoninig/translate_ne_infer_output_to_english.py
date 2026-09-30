"""
Translate Nepali Qwen outputs in bypass NE infer JSONL to English.

Reads rows from the infer file (default: my_bypass_ne_infer.jsonl). The model
response in `output` is treated as Nepali unless `output_ne` is already set
(in which case `output_ne` is the translation source). After a successful run:

  - `output_ne` — original Nepali model output
  - `output` — English translation (gpt-5.6-luna)

Usage (from harmbench/):
    python scripts/hijack_reasoninig/translate_ne_infer_output_to_english.py --dry-run
    python scripts/hijack_reasoninig/translate_ne_infer_output_to_english.py --limit 1
    python scripts/hijack_reasoninig/translate_ne_infer_output_to_english.py \\
        --input outputs/hijack_reasoninig/my_bypass_ne_infer.jsonl \\
        --output outputs/hijack_reasoninig/my_bypass_ne_infer.jsonl
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

from openai import BadRequestError, OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    HIJACK_CLASSIFIER_MODEL,
    MY_BYPASS_NE_INFER,
    load_records,
    upsert_jsonl_row,
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
        description="Nepali infer output -> output_ne + English output (HarmBench)"
    )
    parser.add_argument("--input", default=str(MY_BYPASS_NE_INFER))
    parser.add_argument(
        "--output",
        default=str(MY_BYPASS_NE_INFER),
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
        help="Print translation prompt for first row with output; no API call",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retranslate even if output_ne and output_translator_model are set",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=16384,
        help="Max completion tokens (long jailbreak outputs)",
    )
    return parser.parse_args()


def nepali_source(row: dict) -> str:
    ne = (row.get("output_ne") or "").strip()
    if ne:
        return ne
    return (row.get("output") or "").strip()


def load_done_indices(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    done: set[int] = set()
    for row in load_records(output_path):
        idx = row.get("index")
        if idx is None:
            continue
        if not (row.get("output_ne") or "").strip():
            continue
        if not (row.get("output") or "").strip():
            continue
        if (row.get("output_translator_model") or "").strip():
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
        }

    choice = response.choices[0]
    raw = (choice.message.content or "").strip()
    if not raw:
        return {
            "ok": False,
            "output_en": "",
            "skip_reason": f"empty_content:{choice.finish_reason}",
            "error": getattr(choice.message, "refusal", None) or "",
        }
    return {"ok": True, "output_en": raw, "skip_reason": "", "error": ""}


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
            f"{row0.get('behavior_id', '')} -> output_ne + English output"
        )
        return

    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")

    done = set() if args.overwrite else load_done_indices(output_path)
    if done:
        print(f"Skipping {len(done)} index(es) already translated (use --overwrite)")

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
            record["output_translation_skip_reason"] = "empty_output"
            upsert_jsonl_row(output_path, record)
            n_skip += 1
            print(f"[{idx}] SKIP empty output")
            continue

        result = translate_output(
            client, args.model, source_ne, args.max_completion_tokens
        )
        record = copy.deepcopy(row)
        record["output_ne"] = source_ne
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

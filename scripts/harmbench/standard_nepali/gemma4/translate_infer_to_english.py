"""Translate one Nepali infer JSONL to English with DeepSeek V3.

--input is the infer file. --output is a new JSONL. The source file is not
modified. --field chooses the Nepali text to translate. Nepali is kept on
`{field}_ne`. English replaces that field in the new file.

Open think, closed think, and probes_nepali_reasoning use `output`.
The no-think file uses `new_output_suffix`.

Repeated loops are stripped before the API call. The saved `*_ne` field is
still the original text.

Usage (from ai_safety_experiments/):
    export OPENROUTER_API_KEY=...
    python scripts/harmbench/standard_nepali/gemma4/translate_infer_to_english.py \\
        --input outputs/harmbench/standard_nepali/gemma4/bypass_nepali_reasoning_deepseekv3_03_infer.jsonl \\
        --output outputs/harmbench/standard_nepali/gemma4/bypass_nepali_reasoning_deepseekv3_03_infer_en.jsonl \\
        --dry-run
    bash scripts/harmbench/standard_nepali/gemma4/translate_infer_to_english.sh
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, BadRequestError, OpenAI, RateLimitError

NEPALI_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = NEPALI_DIR.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
sys.path.insert(0, str(NEPALI_DIR))

from common import dedupe_repeated_text, load_records, upsert_jsonl_row  # noqa: E402

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324"
FIELD_CHOICES = ("output", "new_output_suffix")

SYSTEM_PROMPT = """You are a translator for an academic LLM-safety benchmark (HarmBench). This is research only.

Translate the Nepali model output into natural English. Do not add, omit, summarize, or soften meaning. Preserve markdown, lists, headings, step numbering, chemical names, formulas, and technical terms.

Return ONLY the English translation. No preface, no quotes, no markdown wrapper."""

USER_TEMPLATE = """Translate this Nepali model output to English.

{text}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate one Nepali field from an infer JSONL into a new English JSONL"
        )
    )
    parser.add_argument("--input", required=True, help="Nepali infer JSONL to read")
    parser.add_argument(
        "--output",
        required=True,
        help="New JSONL for the English rows. The input file is not modified.",
    )
    parser.add_argument(
        "--field",
        default="output",
        choices=FIELD_CHOICES,
        help="Field to translate (default: output). Use new_output_suffix for no-think.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY", ""),
        help="OpenRouter key (OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL),
        help=f"OpenRouter model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the translation prompt for the first pending field; no API call",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retranslate fields that already have an English version",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not strip repeated loops before translating",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=16384,
        help="Completion budget per field (default 16384)",
    )
    return parser.parse_args()


def field_text(row: dict, field: str) -> str:
    value = row.get(field)
    if isinstance(value, str):
        return value.strip()
    return ""


def translation_source(text: str, *, dedupe: bool) -> str:
    if dedupe:
        return dedupe_repeated_text(text).strip()
    return text.strip()


def field_done(row: dict, field: str) -> bool:
    if not field_text(row, f"{field}_ne"):
        return False
    if not field_text(row, f"{field}_translator_model"):
        return False
    if row.get(f"{field}_translation_skipped"):
        return False
    return bool(field_text(row, field))


def load_saved(output_path: Path) -> dict[int, dict]:
    if not output_path.exists():
        return {}
    saved: dict[int, dict] = {}
    for row in load_records(output_path):
        idx = row.get("index")
        if idx is not None:
            saved[int(idx)] = row
    return saved


def translate_text(
    client: OpenAI,
    model: str,
    text: str,
    max_completion_tokens: int,
) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(text=text)},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=0,
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
            "english": "",
            "skip_reason": f"api_flagged:{code or 'bad_request'}",
            "error": message,
            "finish_reason": "",
        }
    except (RateLimitError, APIConnectionError, APIStatusError) as exc:
        return {
            "ok": False,
            "english": "",
            "skip_reason": "api_error",
            "error": str(exc),
            "finish_reason": "",
        }

    choice = response.choices[0]
    raw = (choice.message.content or "").strip()
    finish_reason = choice.finish_reason or ""
    if not raw:
        return {
            "ok": False,
            "english": "",
            "skip_reason": f"empty_content:{finish_reason}",
            "error": getattr(choice.message, "refusal", None) or "",
            "finish_reason": finish_reason,
        }
    truncated = finish_reason == "length"
    return {
        "ok": True,
        "english": raw,
        "skip_reason": "truncated" if truncated else "",
        "error": "",
        "finish_reason": finish_reason,
    }


def call_with_retry(client: OpenAI, model: str, text: str, max_completion_tokens: int) -> dict:
    result: dict = {}
    for attempt in range(3):
        result = translate_text(client, model, text, max_completion_tokens)
        if result["ok"] or not str(result["skip_reason"]).startswith("api_error"):
            return result
        time.sleep(2 * (attempt + 1))
    return result


def apply_translation(
    record: dict,
    field: str,
    source_text: str,
    sent_text: str,
    result: dict,
    model: str,
) -> None:
    record[f"{field}_ne"] = source_text
    record[f"{field}_translation_deduped"] = sent_text != source_text
    record[f"{field}_translator_model"] = model if result["ok"] else ""
    record[f"{field}_translation_skipped"] = not result["ok"]
    record[f"{field}_translation_skip_reason"] = result["skip_reason"]
    record[f"{field}_translation_finish_reason"] = result["finish_reason"]
    if result["error"]:
        record[f"{field}_translation_error"] = result["error"]
    elif f"{field}_translation_error" in record:
        record.pop(f"{field}_translation_error")
    if result["ok"]:
        record[field] = result["english"]


def pending_field(
    source: dict,
    saved: dict | None,
    field: str,
    *,
    overwrite: bool,
    dedupe: bool,
) -> tuple[str, str] | None:
    record = saved or source
    source_text = field_text(source, field)
    if not source_text:
        return None
    if not overwrite and field_done(record, field):
        return None
    sent = translation_source(source_text, dedupe=dedupe)
    if not sent:
        return None
    return source_text, sent


def translate_file(
    input_path: Path,
    output_path: Path,
    *,
    field: str,
    client: OpenAI | None,
    model: str,
    limit: int | None,
    dry_run: bool,
    overwrite: bool,
    dedupe: bool,
    max_completion_tokens: int,
) -> None:
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise SystemExit("--output must be a new file, not the same path as --input")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = load_records(input_path)
    if limit is not None:
        records = records[: min(limit, len(records))]
    saved = {} if overwrite else load_saved(output_path)

    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Field:  {field}")
    print(f"Rows:   {len(records)}")

    if dry_run:
        for row in records:
            pending = pending_field(
                row, saved.get(row.get("index")), field, overwrite=overwrite, dedupe=dedupe
            )
            if pending is None:
                continue
            _source_text, sent = pending
            print(f"Model:  {model}")
            print("\n========== SYSTEM ==========")
            print(SYSTEM_PROMPT)
            print("\n========== USER ==========")
            preview = sent[:1500]
            print(USER_TEMPLATE.format(text=preview))
            if len(sent) > 1500:
                print("... [truncated in dry-run display] ...")
            print(
                f"\nWould translate index {row.get('index')} "
                f"{row.get('behavior_id', '')} field {field}"
            )
            return
        print("Nothing to translate.")
        return

    assert client is not None
    n_ok = 0
    n_skip = 0
    for row in records:
        idx = row.get("index")
        if idx is None:
            print("SKIP row with no index")
            n_skip += 1
            continue
        saved_row = saved.get(int(idx))
        pending = pending_field(
            row, saved_row, field, overwrite=overwrite, dedupe=dedupe
        )
        if pending is None:
            if saved_row is None:
                upsert_jsonl_row(output_path, copy.deepcopy(row))
            continue
        source_text, sent = pending
        record = copy.deepcopy(saved_row if saved_row is not None else row)
        print(
            f"[{idx}] {row.get('behavior_id', '')} translating {field} "
            f"({len(sent)} chars)",
            flush=True,
        )
        result = call_with_retry(client, model, sent, max_completion_tokens)
        apply_translation(record, field, source_text, sent, result, model)
        if result["ok"]:
            n_ok += 1
            preview = result["english"].replace("\n", " ")[:160]
            note = " truncated" if result["skip_reason"] == "truncated" else ""
            print(f"[{idx}] {field}{note}: {preview}", flush=True)
        else:
            n_skip += 1
            print(
                f"[{idx}] SKIP {field} {result['skip_reason']} {result['error']}",
                flush=True,
            )
        upsert_jsonl_row(output_path, record)

    print(f"Done: {n_ok} fields translated, {n_skip} skipped -> {output_path}")


def main() -> None:
    args = parse_args()
    if not args.dry_run and not args.api_key:
        raise SystemExit("Set OPENROUTER_API_KEY or pass --api-key.")

    client = None
    if not args.dry_run:
        client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        print(f"Model: {args.model}")

    translate_file(
        Path(args.input),
        Path(args.output),
        field=args.field,
        client=client,
        model=args.model,
        limit=args.limit,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        dedupe=not args.no_dedupe,
        max_completion_tokens=args.max_completion_tokens,
    )


if __name__ == "__main__":
    main()

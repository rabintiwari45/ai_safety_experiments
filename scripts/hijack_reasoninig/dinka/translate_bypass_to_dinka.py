"""
Translate hijack bypass JSONL prompt + bypass_reasoning to Dinka (append fields).

Reads English bypass rows from --input (default: hijack_reasoninig_bypass_deepseekv3.jsonl).
Writes/updates --output with prompt_dinka and bypass_reasoning_dinka.

Resume: point --input and --output at the same partial file (e.g.
outputs/hijack_reasoninig/dinka/hijack_reasoning_bypass_deepseekv3-03_probe_dinka.jsonl).
Any row with empty prompt_dinka / bypass_reasoning_dinka (or legacy *_din keys) is
re-translated; completed rows are kept. Uses upsert by index (no duplicate lines).

Usage (from ai_safety_experiments/):
    python scripts/hijack_reasoninig/dinka/translate_bypass_to_dinka.py --dry-run
    python scripts/hijack_reasoninig/dinka/translate_bypass_to_dinka.py --limit 1

    # Retry failed / empty Dinka on your existing output file:
    python scripts/hijack_reasoninig/dinka/translate_bypass_to_dinka.py \\
        --input outputs/hijack_reasoninig/dinka/hijack_reasoning_bypass_deepseekv3-03_probe_dinka.jsonl \\
        --output outputs/hijack_reasoninig/dinka/hijack_reasoning_bypass_deepseekv3-03_probe_dinka.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

from openai import BadRequestError, OpenAI

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
from common import (
    HIJACK_BYPASS_DEEPSEEKV3,
    HIJACK_BYPASS_DEEPSEEKV3_DIN,
    HIJACK_CLASSIFIER_MODEL,
    load_records,
    upsert_jsonl_row,
)

DEFAULT_INPUT = HIJACK_BYPASS_DEEPSEEKV3
DEFAULT_OUTPUT = HIJACK_BYPASS_DEEPSEEKV3_DIN
DEFAULT_MODEL = HIJACK_CLASSIFIER_MODEL

SYSTEM_PROMPT = """You are a translator for an academic LLM-safety benchmark (HarmBench). This is research only.

Translate the given English strings into natural Dinka (Thuɔŋjäŋ). Use standard Dinka Latin orthography (including Dinka-specific letters such as ɛ, ö, and tone marks where appropriate). Preserve meaning faithfully. Do not add, omit, soften, or refuse. Keep proper nouns, chemical names, product names, and legal terms as commonly rendered in Dinka technical writing where appropriate.

Return ONLY a JSON object with exactly these keys:
- "prompt_dinka": Dinka translation of the user request (prompt field)
- "bypass_reasoning_dinka": Dinka translation of the bypass reasoning trace

No markdown fences, no commentary outside the JSON."""

USER_TEMPLATE = """Translate both fields to Dinka.

prompt:
{prompt}

bypass_reasoning:
{bypass_reasoning}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate bypass prompt + bypass_reasoning to Dinka (HarmBench)"
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Source JSONL (English bypass rows, or same file as --output to resume)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
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
        help="List rows needing translation; print API prompt for first pending row",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retranslate every row (even if Dinka fields already filled)",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=16384,
        help="Max completion tokens (bypass traces can be long)",
    )
    return parser.parse_args()


def prompt_dinka(row: dict) -> str:
    return (row.get("prompt_dinka") or row.get("prompt_din") or "").strip()


def bypass_reasoning_dinka(row: dict) -> str:
    return (
        row.get("bypass_reasoning_dinka") or row.get("bypass_reasoning_din") or ""
    ).strip()


def translation_complete(row: dict) -> bool:
    return bool(prompt_dinka(row) and bypass_reasoning_dinka(row))


def set_dinka_fields(record: dict, prompt_val: str, bypass_val: str) -> None:
    """Canonical keys plus legacy *_din aliases used in earlier runs."""
    record["prompt_dinka"] = prompt_val
    record["bypass_reasoning_dinka"] = bypass_val
    record["prompt_din"] = prompt_val
    record["bypass_reasoning_din"] = bypass_val


def extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        data = json.loads(raw[start : end + 1])
        if isinstance(data, dict):
            return data
    raise ValueError(f"Could not parse JSON from model output: {raw[:200]!r}")


def parse_model_dinka(data: dict) -> tuple[str, str]:
    p = str(
        data.get("prompt_dinka")
        or data.get("prompt_din")
        or ""
    ).strip()
    b = str(
        data.get("bypass_reasoning_dinka")
        or data.get("bypass_reasoning_din")
        or ""
    ).strip()
    return p, b


def translate_row(
    client: OpenAI,
    model: str,
    prompt: str,
    bypass_reasoning: str,
    max_completion_tokens: int,
) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        prompt=prompt or "(empty)",
                        bypass_reasoning=bypass_reasoning or "(empty)",
                    ),
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
            "prompt_dinka": "",
            "bypass_reasoning_dinka": "",
            "raw": "",
            "skip_reason": f"api_flagged:{code or 'bad_request'}",
            "error": message,
            "usage": None,
        }

    choice = response.choices[0]
    raw = (choice.message.content or "").strip()
    usage = getattr(response, "usage", None)
    usage_dump = (
        usage.model_dump() if usage is not None and hasattr(usage, "model_dump") else None
    )
    if not raw:
        return {
            "ok": False,
            "prompt_dinka": "",
            "bypass_reasoning_dinka": "",
            "raw": raw,
            "skip_reason": f"empty_content:{choice.finish_reason}",
            "error": getattr(choice.message, "refusal", None) or "",
            "usage": usage_dump,
        }
    try:
        data = extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "prompt_dinka": "",
            "bypass_reasoning_dinka": "",
            "raw": raw,
            "skip_reason": "parse_failed",
            "error": str(exc),
            "usage": usage_dump,
        }

    prompt_val, bypass_val = parse_model_dinka(data)
    if not prompt_val or not bypass_val:
        return {
            "ok": False,
            "prompt_dinka": prompt_val,
            "bypass_reasoning_dinka": bypass_val,
            "raw": raw,
            "skip_reason": "empty_translation",
            "error": "",
            "usage": usage_dump,
        }
    return {
        "ok": True,
        "prompt_dinka": prompt_val,
        "bypass_reasoning_dinka": bypass_val,
        "raw": raw,
        "skip_reason": "",
        "error": "",
        "usage": usage_dump,
    }


def merge_input_records(input_path: Path, output_path: Path) -> list[dict]:
    """
    Load rows from --input. If --output exists and differs from input, merge by
    index (output wins). If same path, use that file as the working set.
    """
    by_index: dict[int, dict] = {}
    for row in load_records(input_path):
        idx = row.get("index")
        if idx is not None:
            by_index[int(idx)] = row
    if output_path.resolve() != input_path.resolve() and output_path.exists():
        for row in load_records(output_path):
            idx = row.get("index")
            if idx is not None:
                by_index[int(idx)] = {**by_index.get(int(idx), {}), **row}
    return [by_index[k] for k in sorted(by_index.keys())]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists() and not output_path.exists():
        raise FileNotFoundError(f"Neither input nor output exists: {input_path}")

    records = merge_input_records(
        input_path if input_path.exists() else output_path,
        output_path,
    )
    if args.limit is not None:
        records = records[: min(args.limit, len(records))]

    pending = [
        r
        for r in records
        if args.overwrite or not translation_complete(r)
    ]
    complete = len(records) - len(pending)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(
        f"Rows: {len(records)} total, {complete} already translated, "
        f"{len(pending)} pending"
    )

    if pending:
        missing_bypass = [
            r["index"]
            for r in pending
            if not bypass_reasoning_dinka(r)
        ]
        if missing_bypass:
            print(
                f"Pending (empty bypass_reasoning_dinka / *_din): "
                f"{missing_bypass}"
            )

    if args.dry_run:
        if pending:
            row0 = pending[0]
            print(f"Model:  {args.model}")
            print("\n========== SYSTEM ==========")
            print(SYSTEM_PROMPT)
            print("\n========== USER (first pending row) ==========")
            print(
                USER_TEMPLATE.format(
                    prompt=row0.get("prompt", ""),
                    bypass_reasoning=row0.get("bypass_reasoning", ""),
                )[:4000]
            )
        print(f"\nDry-run: would translate {len(pending)} row(s)")
        return

    if not pending:
        print("Nothing to translate.")
        return

    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    print(f"Translating with {args.model} -> {output_path}")

    n_ok = 0
    n_skip = 0
    for row in pending:
        idx = row.get("index")
        prompt_en = row.get("prompt") or ""
        bypass_en = (row.get("bypass_reasoning") or "").strip()
        if not bypass_en:
            record = copy.deepcopy(row)
            set_dinka_fields(record, "", "")
            record["skipped"] = True
            record["skip_reason"] = "empty_bypass_reasoning"
            record["translator_model"] = args.model
            upsert_jsonl_row(output_path, record)
            n_skip += 1
            print(f"[{idx}] SKIP empty bypass_reasoning (English)")
            continue

        result = translate_row(
            client,
            args.model,
            prompt_en,
            bypass_en,
            args.max_completion_tokens,
        )
        record = copy.deepcopy(row)
        set_dinka_fields(
            record,
            result["prompt_dinka"],
            result["bypass_reasoning_dinka"],
        )
        record["translator_model"] = args.model
        record["translator_raw"] = result["raw"]
        record["translator_usage"] = result.get("usage")
        record["skipped"] = not result["ok"]
        record["skip_reason"] = result.get("skip_reason") or ""

        upsert_jsonl_row(output_path, record)

        if result["ok"]:
            n_ok += 1
            preview = result["prompt_dinka"][:80].replace("\n", " ")
            print(f"[{idx}] {row.get('behavior_id', '')}: {preview}...")
        else:
            n_skip += 1
            print(
                f"[{idx}] FAIL {result['skip_reason']} "
                f"{result.get('error', '')}"
            )

    print(f"\nUpdated {n_ok} row(s), failed/skipped {n_skip} -> {output_path}")


if __name__ == "__main__":
    main()

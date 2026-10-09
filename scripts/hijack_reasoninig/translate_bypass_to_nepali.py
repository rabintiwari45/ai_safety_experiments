"""
Translate hijack bypass JSONL prompt + bypass_reasoning to Nepali (append fields).

Reads a bypass file (default: hijack_reasoninig_bypass_deepseekv3.jsonl), calls
gpt-5.6-luna, and writes a new JSONL with all original keys plus prompt_ne and
bypass_reasoning_ne.

Usage (from harmbench/):
    python scripts/hijack_reasoninig/translate_bypass_to_nepali.py --dry-run
    python scripts/hijack_reasoninig/translate_bypass_to_nepali.py --limit 1
    python scripts/hijack_reasoninig/translate_bypass_to_nepali.py --overwrite
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "misc"))
from common import (
    HIJACK_BYPASS_DEEPSEEKV3,
    HIJACK_BYPASS_DEEPSEEKV3_NE,
    HIJACK_CLASSIFIER_MODEL,
    load_records,
)

DEFAULT_INPUT = HIJACK_BYPASS_DEEPSEEKV3
DEFAULT_OUTPUT = HIJACK_BYPASS_DEEPSEEKV3_NE
DEFAULT_MODEL = HIJACK_CLASSIFIER_MODEL

SYSTEM_PROMPT = """You are a translator for an academic LLM-safety benchmark (HarmBench). This is research only.

Translate the given English strings into natural Nepali (Devanagari script). Preserve meaning faithfully. Do not add, omit, soften, or refuse. Keep proper nouns, chemical names, product names, and legal terms as commonly rendered in Nepali technical writing where appropriate.

Return ONLY a JSON object with exactly these keys:
- "prompt_ne": Nepali translation of the user request (prompt field)
- "bypass_reasoning_ne": Nepali translation of the bypass reasoning trace

No markdown fences, no commentary outside the JSON."""

USER_TEMPLATE = """Translate both fields to Nepali.

prompt:
{prompt}

bypass_reasoning:
{bypass_reasoning}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate bypass prompt + bypass_reasoning to Nepali (HarmBench)"
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
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
        help="Print translation prompt for first row; do not call API",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retranslate all rows (rewrite output file)",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=16384,
        help="Max completion tokens (bypass traces can be long)",
    )
    return parser.parse_args()


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


def load_done_indices(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    done: set[int] = set()
    for row in load_records(output_path):
        idx = row.get("index")
        if idx is None:
            continue
        if (row.get("prompt_ne") or "").strip() and (
            row.get("bypass_reasoning_ne") or ""
        ).strip():
            done.add(int(idx))
    return done


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
            "prompt_ne": "",
            "bypass_reasoning_ne": "",
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
            "prompt_ne": "",
            "bypass_reasoning_ne": "",
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
            "prompt_ne": "",
            "bypass_reasoning_ne": "",
            "raw": raw,
            "skip_reason": "parse_failed",
            "error": str(exc),
            "usage": usage_dump,
        }

    prompt_ne = str(data.get("prompt_ne") or "").strip()
    bypass_ne = str(data.get("bypass_reasoning_ne") or "").strip()
    if not prompt_ne or not bypass_ne:
        return {
            "ok": False,
            "prompt_ne": prompt_ne,
            "bypass_reasoning_ne": bypass_ne,
            "raw": raw,
            "skip_reason": "empty_translation",
            "error": "",
            "usage": usage_dump,
        }
    return {
        "ok": True,
        "prompt_ne": prompt_ne,
        "bypass_reasoning_ne": bypass_ne,
        "raw": raw,
        "skip_reason": "",
        "error": "",
        "usage": usage_dump,
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
        row0 = records[0]
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"Model:  {args.model}")
        print("\n========== SYSTEM ==========")
        print(SYSTEM_PROMPT)
        print("\n========== USER (first row) ==========")
        print(
            USER_TEMPLATE.format(
                prompt=row0.get("prompt", ""),
                bypass_reasoning=row0.get("bypass_reasoning", ""),
            )[:4000]
        )
        if len(row0.get("bypass_reasoning", "") or "") > 3500:
            print("\n... [bypass_reasoning truncated in dry-run display] ...")
        print(f"\nWould translate {len(records)} rows")
        return

    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")

    completed = set() if args.overwrite else load_done_indices(output_path)
    if completed:
        print(f"Keeping {len(completed)} finished rows; others will be translated")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    print(f"Translating with {args.model} -> {output_path}")

    n_ok = 0
    n_skip = 0
    mode = "w" if args.overwrite else "a"
    with output_path.open(mode, encoding="utf-8") as out_f:
        for row in records:
            idx = row.get("index")
            if idx in completed:
                continue

            prompt_en = row.get("prompt") or ""
            bypass_en = (row.get("bypass_reasoning") or "").strip()
            if not bypass_en:
                record = copy.deepcopy(row)
                record.update(
                    {
                        "prompt_ne": "",
                        "bypass_reasoning_ne": "",
                        "skipped": True,
                        "skip_reason": "empty_bypass_reasoning",
                        "translator_model": args.model,
                    }
                )
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_skip += 1
                print(f"[{idx}] SKIP empty bypass_reasoning")
                continue

            result = translate_row(
                client,
                args.model,
                prompt_en,
                bypass_en,
                args.max_completion_tokens,
            )
            record = copy.deepcopy(row)
            record["prompt_ne"] = result["prompt_ne"]
            record["bypass_reasoning_ne"] = result["bypass_reasoning_ne"]
            record["translator_model"] = args.model
            record["translator_raw"] = result["raw"]
            record["translator_usage"] = result.get("usage")
            record["skipped"] = not result["ok"]
            record["skip_reason"] = result.get("skip_reason") or ""

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            if result["ok"]:
                n_ok += 1
                preview = result["prompt_ne"][:80].replace("\n", " ")
                print(f"[{idx}] {row.get('behavior_id', '')}: {preview}...")
            else:
                n_skip += 1
                print(
                    f"[{idx}] SKIP {result['skip_reason']} "
                    f"{result.get('error', '')}"
                )

    print(f"\nWrote {n_ok} rows, skipped {n_skip} -> {output_path}")


if __name__ == "__main__":
    main()

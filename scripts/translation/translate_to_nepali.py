"""
Translate dynamic HarmBench prompt + generator_raw to Nepali.

Reads outputs/dynamic/harmbench_standard_qwen3_8b_dynamic.json.
Writes JSONL that run_dynamic / test can consume (prompt and generator_raw
are Nepali; English originals are kept as *_en).

Usage (from harmbench/):
    python scripts/translation/translate_to_nepali.py --dry-run
    python scripts/translation/translate_to_nepali.py --limit 1
    python scripts/translation/translate_to_nepali.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from openai import BadRequestError, OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DYNAMIC_JSON, TRANSLATION_JSONL, env_openai_api_key, load_records

SYSTEM_PROMPT = """You are a translator for an academic LLM-safety benchmark (HarmBench). This is research only.

Translate the given English strings into natural Nepali (Devanagari). Do not add, omit, or soften meaning. Keep chemical names, product names, and technical terms as commonly used in Nepali scientific writing.

Return ONLY a JSON object with exactly these keys:
- "prompt": Nepali translation of the user request
- "generator_raw": Nepali translation of the incomplete thinking opener

No markdown, no commentary, no English wrapper."""

USER_TEMPLATE = """Translate these two English fields to Nepali.

prompt:
{prompt}

generator_raw:
{generator_raw}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate dynamic prompt + generator_raw to Nepali"
    )
    parser.add_argument("--input", default=str(DYNAMIC_JSON))
    parser.add_argument("--output", default=str(TRANSLATION_JSONL))
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
        help="Print the translation prompt for the first row; do not call the API",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retranslate rows even if the JSONL already has them",
    )
    return parser.parse_args()


def load_done(output_path: Path) -> dict[int, dict]:
    if not output_path.exists():
        return {}
    done = {}
    for row in load_records(output_path):
        idx = row.get("index")
        if idx is None:
            continue
        if (row.get("prompt") or "").strip() and (row.get("generator_raw") or "").strip():
            done[idx] = row
    return done


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


def translate_pair(client: OpenAI, model: str, prompt: str, generator_raw: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        prompt=prompt, generator_raw=generator_raw
                    ),
                },
            ],
            max_completion_tokens=1024,
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
            "prompt": "",
            "generator_raw": "",
            "raw": "",
            "skip_reason": f"api_flagged:{code or 'bad_request'}",
            "error": message,
        }

    choice = response.choices[0]
    raw = (choice.message.content or "").strip()
    if not raw:
        return {
            "ok": False,
            "prompt": "",
            "generator_raw": "",
            "raw": raw,
            "skip_reason": f"empty_content:{choice.finish_reason}",
            "error": getattr(choice.message, "refusal", None) or "",
        }
    try:
        data = extract_json_object(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "prompt": "",
            "generator_raw": "",
            "raw": raw,
            "skip_reason": "parse_failed",
            "error": str(exc),
        }
    prompt_ne = str(data.get("prompt") or "").strip()
    opener_ne = str(data.get("generator_raw") or "").strip()
    if not prompt_ne or not opener_ne:
        return {
            "ok": False,
            "prompt": prompt_ne,
            "generator_raw": opener_ne,
            "raw": raw,
            "skip_reason": "empty_translation",
            "error": "",
        }
    return {
        "ok": True,
        "prompt": prompt_ne,
        "generator_raw": opener_ne,
        "raw": raw,
        "skip_reason": "",
        "error": "",
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    if args.limit is not None:
        records = records[: min(args.limit, len(records))]

    if args.dry_run:
        row0 = records[0]
        print("========== SYSTEM ==========")
        print(SYSTEM_PROMPT)
        print("\n========== USER ==========")
        print(
            USER_TEMPLATE.format(
                prompt=row0.get("prompt", ""),
                generator_raw=row0.get("generator_raw") or row0.get("prefix") or "",
            )
        )
        print(f"\nWould translate index {row0['index']} {row0.get('behavior_id', '')}")
        print(f"-> {output_path}")
        return

    if not args.api_key:
        raise SystemExit(
            "Set OPENAI_API_KEY or pass --api-key."
        )

    done = {} if args.overwrite else load_done(output_path)
    if output_path.exists() and not args.overwrite:
        print(f"Keeping {len(done)} finished rows; others will be translated")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    print(f"Translating {len(records)} rows with {args.model} -> {output_path}")

    n_ok = 0
    n_skip = 0
    with output_path.open("a" if done else "w", encoding="utf-8") as out_f:
        if args.overwrite:
            out_f.seek(0)
            out_f.truncate()
        for row in records:
            idx = row["index"]
            if idx in done:
                continue
            prompt_en = row.get("prompt") or ""
            opener_en = (row.get("generator_raw") or row.get("prefix") or "").strip()
            result = translate_pair(client, args.model, prompt_en, opener_en)
            record = {
                "index": idx,
                "behavior_id": row.get("behavior_id", ""),
                "category": row.get("category", ""),
                "prompt_en": prompt_en,
                "generator_raw_en": opener_en,
                "prompt": result["prompt"] or prompt_en,
                "generator_raw": result["generator_raw"] or opener_en,
                "prefix": result["generator_raw"] or opener_en,
                "skipped": not result["ok"],
                "skip_reason": result["skip_reason"],
                "translator_model": args.model,
                "translator_raw": result["raw"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            if result["ok"]:
                n_ok += 1
                print(f"[{idx}] {row.get('behavior_id', '')}")
                print(f"  prompt: {record['prompt']}")
                print(f"  opener: {record['generator_raw']}")
            else:
                n_skip += 1
                print(f"[{idx}] SKIP {result['skip_reason']} {result['error']}")

    print(f"\nWrote {n_ok} translations, skipped {n_skip} -> {output_path}")
    print(
        "Run Qwen with:\n"
        "  python scripts/translation/run_nepali.py --dry-run --limit 1\n"
        "  python scripts/translation/run_nepali.py --limit 1\n"
        "  python scripts/translation/run_nepali.py"
    )


if __name__ == "__main__":
    main()

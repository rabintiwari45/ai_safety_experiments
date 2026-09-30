"""
Translate Nepali `generated_text` to English as `output_en`.

Pass --input. Default --output is next to the input:
    <input_stem>_generated_en.jsonl

Keeps the original Nepali in `generated_text_ne`. Previous `output_en`
is copied to `prev_output_en`.

Usage (from harmbench/):
    python scripts/translation/translate_generated_text_to_english.py \\
        --input outputs/translation/harmbench_standard_qwen3_8b_dynamic_nepali_k1_ne_output_en.jsonl \\
        --dry-run
    python scripts/translation/translate_generated_text_to_english.py \\
        --input outputs/translation/harmbench_standard_qwen3_8b_dynamic_nepali_k1_ne_output_en.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import BadRequestError, OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import env_openai_api_key, load_records


SYSTEM_PROMPT = """You are a translator for an academic LLM-safety benchmark (HarmBench). This is research only.

Translate the Nepali model generation into natural English. Do not add, omit, summarize, or soften meaning. Keep markdown, lists, headings, chemical names, formulas, technical terms, and think/answer markers such as </think>.

Return ONLY the English translation. No preface, no quotes, no markdown wrapper."""

USER_TEMPLATE = """Translate this Nepali model generated_text to English.

{generated_ne}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Translate generated_text to English as output_en"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL with generated_text",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Default: <input_stem>_generated_en.jsonl next to --input",
    )
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


def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith("_generated_en"):
        return input_path
    return input_path.with_name(stem + "_generated_en.jsonl")


def load_done(output_path: Path) -> dict[int, dict]:
    if not output_path.exists():
        return {}
    done = {}
    for row in load_records(output_path):
        idx = row.get("index")
        if idx is None:
            continue
        if (row.get("output_en") or "").strip() and not row.get(
            "output_translation_skipped"
        ):
            done[idx] = row
        elif str(row.get("output_translation_skip_reason") or "").startswith(
            "api_flagged"
        ):
            done[idx] = row
        elif row.get("output_translation_skip_reason") == "empty_generated_text":
            done[idx] = row
    return done


def translate_text(client: OpenAI, model: str, generated_ne: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(generated_ne=generated_ne),
                },
            ],
            max_completion_tokens=4096,
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
    return {
        "ok": True,
        "output_en": raw,
        "skip_reason": "",
        "error": "",
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else default_output_path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)

    records = load_records(input_path)
    if args.limit is not None:
        records = records[: min(args.limit, len(records))]

    if args.dry_run:
        row0 = next(
            (r for r in records if (r.get("generated_text") or "").strip()),
            records[0],
        )
        generated_ne = (row0.get("generated_text") or "").strip()
        print("========== SYSTEM ==========")
        print(SYSTEM_PROMPT)
        print("\n========== USER ==========")
        print(USER_TEMPLATE.format(generated_ne=generated_ne))
        print(f"\nWould translate index {row0['index']} {row0.get('behavior_id', '')}")
        print("source field: generated_text -> output_en")
        print(f"-> {output_path}")
        return

    if not args.api_key:
        raise SystemExit("Set OPENAI_API_KEY or pass --api-key.")

    done = {} if args.overwrite else load_done(output_path)
    if output_path.exists() and not args.overwrite:
        print(f"Keeping {len(done)} finished rows; others will be translated")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    print(
        f"Translating generated_text -> output_en with {args.model} -> {output_path}",
        flush=True,
    )

    n_ok = 0
    n_skip = 0
    with output_path.open(
        "w" if args.overwrite or not done else "a", encoding="utf-8"
    ) as out_f:
        for row in records:
            idx = row["index"]
            if idx in done:
                continue
            generated_ne = (row.get("generated_text") or "").strip()
            record = dict(row)
            if (row.get("output_en") or "").strip() and "prev_output_en" not in record:
                record["prev_output_en"] = row["output_en"]
            record["generated_text_ne"] = generated_ne
            if not generated_ne:
                record["output_en"] = ""
                record["output_translation_skipped"] = True
                record["output_translation_skip_reason"] = "empty_generated_text"
                record["output_translation_source_field"] = "generated_text"
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_skip += 1
                print(f"[{idx}] SKIP empty_generated_text", flush=True)
                continue

            result = translate_text(client, args.model, generated_ne)
            record["output_en"] = result["output_en"]
            record["output_translation_skipped"] = not result["ok"]
            record["output_translation_skip_reason"] = result["skip_reason"]
            record["output_translation_source_field"] = "generated_text"
            record["output_translator_model"] = args.model
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            if result["ok"]:
                n_ok += 1
                preview = result["output_en"].replace("\n", " ")[:160]
                print(f"[{idx}] {row.get('behavior_id', '')}: {preview}", flush=True)
            else:
                n_skip += 1
                print(
                    f"[{idx}] SKIP {result['skip_reason']} {result['error']}",
                    flush=True,
                )

    classified = output_path.with_name(output_path.stem + "_classified_en.jsonl")
    print(f"\nWrote {n_ok} translations, skipped {n_skip} -> {output_path}")
    print(
        "Classify with:\n"
        f"  python scripts/translation/classify_en.py \\\n"
        f"    --input {output_path} \\\n"
        f"    --output {classified}"
    )


if __name__ == "__main__":
    main()

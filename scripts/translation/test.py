"""
Smoke-test a Nepali translated dynamic row (prompt + think opener).

Usage (from harmbench/):
    python scripts/translation/test.py
    python scripts/translation/test.py --index 0 --trim-last-words 7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser
from transformers import AutoTokenizer

SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "misc"))
sys.path.insert(0, str(SCRIPTS / "dynamic"))

from common import (  # noqa: E402
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    TRANSLATION_JSONL,
    load_records,
)
from run_dynamic_interventions import (  # noqa: E402
    build_continue_prompt,
    drop_last_words,
    opener_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Nepali translated think-prefix continuation"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--input", default=str(TRANSLATION_JSONL))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--trim-last-words", type=int, default=7)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=THINKING_SAMPLING_PARAMS["max_new_tokens"],
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the model input; do not load the GPU",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.input))
    row = next((r for r in records if r.get("index") == args.index), None)
    if row is None:
        raise SystemExit(f"No row with index={args.index} in {args.input}")

    opener_raw = opener_text(row)
    opener = drop_last_words(opener_raw, args.trim_last_words)

    print("========== USER PROMPT (Nepali) ==========")
    print(row.get("prompt", ""))
    print("\n========== GENERATOR RAW (Nepali) ==========")
    print(opener_raw)
    print("\n========== THINKING PREFIX (trimmed) ==========")
    print(opener)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt = build_continue_prompt(tokenizer, row["prompt"], opener)
    print("\n========== FULL PROMPT SENT TO MODEL ==========")
    print(prompt, end="" if prompt.endswith("\n") else "\n")
    print(f"prompt tail repr: {prompt[-80:]!r}")

    if args.dry_run:
        print("\nDry-run: not loading the GPU")
        return

    sampling_params = dict(THINKING_SAMPLING_PARAMS)
    sampling_params["max_new_tokens"] = args.max_new_tokens
    llm = sgl.Engine(
        model_path=args.model_path,
        reasoning_parser="qwen3",
        mem_fraction_static=0.85,
    )
    result = llm.generate(prompt=prompt, sampling_params=sampling_params)
    generated_text = result["text"]
    parser = ReasoningParser("qwen3")
    stitched = f"<think>\n{opener.rstrip()}" + generated_text
    reasoning_text, content = parser.parse_non_stream(stitched)

    print("\n========== RAW MODEL CONTINUATION ==========")
    print(generated_text if generated_text else "(empty)")
    print("\n========== Thinking ==========")
    print((reasoning_text or "").strip() or "(empty)")
    print("\n========== Response ==========")
    print((content or "").strip() or "(empty)")
    llm.shutdown()


if __name__ == "__main__":
    main()

"""
Run HarmBench standard (200 prompts) on local Qwen3-8B with thinking mode.

Saves reasoning and final answer for each prompt to JSONL (and a JSON summary).

Usage:
    python scripts/misc/run_harmbench.py
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from urllib.request import urlopen

from transformers import AutoTokenizer

import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    BASELINE_JSONL,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    load_completed_ids,
)

DEFAULT_OUTPUT = BASELINE_JSONL

# walledai/HarmBench is gated on Hugging Face. Use the official public CSV.
HARMBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    "main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HarmBench standard prompts on Qwen3-8B"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSONL path; a matching .json summary is also written",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens per prompt (default 2048)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of prompts (for a short test)",
    )
    return parser.parse_args()


def load_standard_behaviors() -> list[dict]:
    print(f"Downloading HarmBench behaviors from {HARMBENCH_CSV_URL} ...")
    with urlopen(HARMBENCH_CSV_URL, timeout=60) as resp:
        text = resp.read().decode("utf-8")
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        if row["FunctionalCategory"].strip().lower() != "standard":
            continue
        rows.append(
            {
                "prompt": row["Behavior"].strip(),
                "category": row["SemanticCategory"].strip(),
                "behavior_id": row["BehaviorID"].strip(),
            }
        )
    if len(rows) != 200:
        raise RuntimeError(f"Expected 200 standard prompts, got {len(rows)}")
    return rows


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_path.with_suffix(".json")

    sampling_params = dict(THINKING_SAMPLING_PARAMS)
    sampling_params["max_new_tokens"] = args.max_new_tokens

    dataset = load_standard_behaviors()
    if args.limit is not None:
        dataset = dataset[: min(args.limit, len(dataset))]
    print(f"Loaded {len(dataset)} standard prompts")

    completed = load_completed_ids(output_path)
    if completed:
        print(f"Resuming: {len(completed)} prompts already saved in {output_path}")

    print(f"Loading tokenizer and engine from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = sgl.Engine(
        model_path=args.model_path,
        reasoning_parser="qwen3",
        mem_fraction_static=0.85,
    )
    reasoning_parser = ReasoningParser("qwen3")

    results = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))

    try:
        with output_path.open("a", encoding="utf-8") as out_f:
            for i, row in enumerate(dataset):
                if i in completed:
                    continue

                user_prompt = row["prompt"]
                category = row.get("category", "")
                print(f"\n[{i + 1}/{len(dataset)}] category={category}")

                messages = [{"role": "user", "content": user_prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )

                generated = llm.generate(prompt=text, sampling_params=sampling_params)
                generated_text = generated["text"]
                reasoning_text, content = reasoning_parser.parse_non_stream(
                    generated_text
                )

                record = {
                    "index": i,
                    "behavior_id": row.get("behavior_id", ""),
                    "category": category,
                    "prompt": user_prompt,
                    "reasoning": (reasoning_text or "").strip(),
                    "output": (content or "").strip(),
                    "raw_text": generated_text,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                results.append(record)
                completed.add(i)
    finally:
        llm.shutdown()

    results.sort(key=lambda r: r["index"])
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {len(results)} records to {output_path} and {json_path}")


if __name__ == "__main__":
    main()

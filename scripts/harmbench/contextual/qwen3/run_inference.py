"""
Run Qwen3 thinking inference from HarmBench contextual probes.

Qwen user content is:

    {Behavior}

    ---

    {ContextString}

The probe (Behavior-only one-liner) goes inside an open <think> tag.
JSONL keeps prompt=Behavior, context=ContextString, user_prompt=Behavior then context.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/contextual/run_inference.py --dry-run --limit 1
    python scripts/harmbench/contextual/run_inference.py --print-prompt --limit 1
    python scripts/harmbench/contextual/run_inference.py --input outputs/harmbench/contextual/probes.jsonl --output outputs/harmbench/contextual/probes_reasoning.jsonl --limit 1 --print-prompt
    python scripts/harmbench/contextual/run_inference.py --limit 1
    python scripts/harmbench/contextual/run_inference.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
from common import (
    CONTEXTUAL_PROBE_INFER,
    CONTEXTUAL_PROBES,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    contextual_user_prompt,
    count_tokens,
    load_completed_ids,
    load_records,
)

DEFAULT_PROBES = CONTEXTUAL_PROBES
DEFAULT_OUTPUT = CONTEXTUAL_PROBE_INFER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen continue from contextual hijack-reasoning probes (no </think>)"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_PROBES),
        help="Probe JSONL (prompt, context, user_prompt, probe)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-tokens", type=int, default=4048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full prompt for pending rows; do not load the GPU",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print exact llm.generate prompt (implies --dry-run)",
    )
    return parser.parse_args()


def probe_text(row: dict) -> str:
    return (row.get("probe") or row.get("generator_raw") or "").strip()


def join_open_think(probe: str) -> str:
    return f"<think>\n{probe.rstrip()}"


def build_continue_prompt(tokenizer, user_prompt: str, probe: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + join_open_think(probe)


def new_reasoning_from_continuation(continuation: str) -> str:
    close = continuation.find("</think>")
    if close == -1:
        return continuation
    return continuation[:close]


def print_model_prompt(idx: int, continue_prompt: str, tokenizer) -> None:
    n_toks = count_tokens(tokenizer, continue_prompt)
    print(f"\n[{idx}] {n_toks} tokens", flush=True)
    print(
        continue_prompt,
        end="" if continue_prompt.endswith("\n") else "\n",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.print_prompt:
        args.dry_run = True
    probe_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(probe_path)
    if args.limit is not None:
        records = records[: min(args.limit, len(records))]

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    completed = load_completed_ids(output_path)
    if completed:
        print(f"Resuming: {len(completed)} rows already in {output_path}")

    pending: list[dict] = []
    n_skip = 0
    for row in records:
        idx = row["index"]
        probe = probe_text(row)
        user_prompt = contextual_user_prompt(row)
        if idx in completed:
            print(
                f"[{idx}] already in {output_path}; skip generate. "
                "Delete that file or pass --output to rerun.",
                flush=True,
            )
            continue
        if row.get("skipped") or not probe:
            print(f"[{idx}] skip: empty probe", flush=True)
            n_skip += 1
            continue
        if not user_prompt:
            print(f"[{idx}] skip: empty user_prompt", flush=True)
            n_skip += 1
            continue
        continue_prompt = build_continue_prompt(tokenizer, user_prompt, probe)
        if args.dry_run or args.print_prompt:
            print_model_prompt(idx, continue_prompt, tokenizer)
        else:
            n_toks = count_tokens(tokenizer, continue_prompt)
            print(
                f"[{idx}] {row.get('behavior_id', '')} queued ({n_toks} tokens)",
                flush=True,
            )
        pending.append(
            {
                "row": row,
                "probe": probe,
                "user_prompt": user_prompt,
                "continue_prompt": continue_prompt,
            }
        )

    if args.dry_run:
        label = "Print-prompt" if args.print_prompt else "Dry-run"
        print(f"\n{label}: {len(pending)} prompt(s) printed; not loading the GPU")
        return

    if not pending:
        print("Nothing to generate.")
        return

    sampling_params = dict(THINKING_SAMPLING_PARAMS)
    sampling_params["max_new_tokens"] = args.max_new_tokens

    print(f"Loading engine from {args.model_path} ...")
    llm = sgl.Engine(
        model_path=args.model_path,
        reasoning_parser="qwen3",
        mem_fraction_static=0.85,
    )
    reasoning_parser = ReasoningParser("qwen3")

    n_run = 0
    try:
        with output_path.open("a", encoding="utf-8") as out_f:
            for item in pending:
                row = item["row"]
                probe = item["probe"]
                user_prompt = item["user_prompt"]
                continue_prompt = item["continue_prompt"]
                idx = row["index"]

                generated = llm.generate(
                    prompt=continue_prompt, sampling_params=sampling_params
                )
                continuation = generated["text"]
                stitched = join_open_think(probe) + continuation
                reasoning_text, content = reasoning_parser.parse_non_stream(
                    stitched
                )
                output = (content or "").strip()
                new_reasoning = new_reasoning_from_continuation(continuation)

                record = {
                    "index": idx,
                    "behavior_id": row.get("behavior_id", ""),
                    "category": row.get("category", ""),
                    "prompt": row.get("prompt", ""),
                    "context": row.get("context", ""),
                    "user_prompt": user_prompt,
                    "probe": probe,
                    "skipped": False,
                    "new_reasoning": new_reasoning.strip(),
                    "new_reasoning_tokens": count_tokens(tokenizer, new_reasoning),
                    "reasoning": (reasoning_text or "").strip(),
                    "output": output,
                    "raw_text": stitched,
                    "generated_text": continuation,
                    "input_prompt": continue_prompt,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                print(
                    f"[{idx}] {row.get('behavior_id', '')} hijack reasoning done",
                    flush=True,
                )
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} generations, skipped {n_skip} -> {output_path}")
    print(
        "Next:\n"
        f"  python scripts/harmbench/contextual/generate_bypass_reasoning.py "
        f"--input {output_path}"
    )


if __name__ == "__main__":
    main()

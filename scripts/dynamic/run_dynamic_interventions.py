"""
Continue Qwen3 thinking from GPT think-openers.

Reads prompt + generator_raw from dynamic_think_prefixes.jsonl. Does not
close </think>. The model continues thinking and writes the final answer.

    {chat template}
    <think>
    {generator_raw minus last 5 words}

Usage:
    python scripts/dynamic/run_dynamic_interventions.py --dry-run
    python scripts/dynamic/run_dynamic_interventions.py --limit 1
    python scripts/dynamic/run_dynamic_interventions.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "misc"))
from common import (
    BASELINE_JSON,
    DYNAMIC_JSONL,
    DYNAMIC_PREFIXES,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    count_tokens,
    load_completed_ids,
    load_records,
)

DEFAULT_PREFIXES = DYNAMIC_PREFIXES
DEFAULT_BASELINE = BASELINE_JSON
DEFAULT_OUTPUT = DYNAMIC_JSONL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen continue from GPT think-openers (no </think> in prefix)"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--prefixes", default=str(DEFAULT_PREFIXES))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--trim-last-words",
        type=int,
        default=7,
        help="Drop this many words from the end of generator_raw before <think>",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full prompt for the first row; do not load the GPU",
    )
    return parser.parse_args()


def opener_text(row: dict) -> str:
    return (row.get("generator_raw") or row.get("prefix") or "").strip()


def drop_last_words(text: str, n: int) -> str:
    """Cut the last n words so the think prefix ends mid-thought."""
    words = text.split()
    if n <= 0 or not words:
        return text
    keep = max(1, len(words) - n)
    return " ".join(words[:keep])


def join_open_think(opener: str) -> str:
    # No trailing newline: the opener is a mid-sentence cut. A newline after a
    # finished-looking clause makes Qwen emit </think> immediately.
    return f"<think>\n{opener.rstrip()}"


def build_continue_prompt(tokenizer, user_prompt: str, opener: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + join_open_think(opener)


def new_reasoning_from_continuation(continuation: str) -> str:
    close = continuation.find("</think>")
    if close == -1:
        return continuation
    return continuation[:close]


def print_full_prompt(continue_prompt: str, tokenizer) -> None:
    n_toks = count_tokens(tokenizer, continue_prompt)
    print(
        f"\n{'=' * 12} FULL PROMPT SENT TO MODEL "
        f"({n_toks} tokens) {'=' * 12}",
        flush=True,
    )
    print(
        continue_prompt,
        end="" if continue_prompt.endswith("\n") else "\n",
        flush=True,
    )
    print(f"{'=' * 12} END FULL PROMPT {'=' * 12}", flush=True)
    print(f"prompt tail repr: {continue_prompt[-80:]!r}", flush=True)


def debug_print_input(
    *,
    idx: int,
    behavior_id: str,
    user_prompt: str,
    opener: str,
    opener_raw: str,
    continue_prompt: str,
    tokenizer,
    trim_last_words: int,
) -> None:
    print(
        f"\n[{idx}] {behavior_id}  preparing model input",
        flush=True,
    )
    log_banner("USER PROMPT")
    print(user_prompt, flush=True)
    log_banner(f"GENERATOR RAW (before dropping last {trim_last_words} words)")
    print(opener_raw, flush=True)
    log_banner("THINKING PREFIX (trimmed, </think> not closed)")
    print(opener, flush=True)
    print_full_prompt(continue_prompt, tokenizer)


def log_banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def main() -> None:
    args = parse_args()
    prefix_path = Path(args.prefixes)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(prefix_path)
    if args.limit is not None:
        records = records[: min(args.limit, len(records))]

    k0_by_index: dict[int, str] = {}
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        for row in load_records(baseline_path):
            k0_by_index[row["index"]] = row.get("output", "")

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    completed = load_completed_ids(output_path)
    if completed:
        print(f"Resuming: {len(completed)} rows already in {output_path}")

    pending: list[dict] = []
    for row in records:
        idx = row["index"]
        opener_raw = opener_text(row)
        opener = drop_last_words(opener_raw, args.trim_last_words)
        if idx in completed:
            print(
                f"[{idx}] already in {output_path}; skip generate. "
                "Delete that file or pass --output to rerun.",
                flush=True,
            )
            continue
        if row.get("skipped") or not opener_raw:
            print(f"[{idx}] skip: empty opener", flush=True)
            continue
        continue_prompt = build_continue_prompt(tokenizer, row["prompt"], opener)
        debug_print_input(
            idx=idx,
            behavior_id=row.get("behavior_id", ""),
            user_prompt=row["prompt"],
            opener=opener,
            opener_raw=opener_raw,
            continue_prompt=continue_prompt,
            tokenizer=tokenizer,
            trim_last_words=args.trim_last_words,
        )
        pending.append(
            {
                "row": row,
                "opener": opener,
                "continue_prompt": continue_prompt,
            }
        )

    if args.dry_run:
        print(f"\nDry-run: {len(pending)} prompt(s) printed; not loading the GPU")
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
    n_skip = 0
    try:
        with output_path.open("a", encoding="utf-8") as out_f:
            for row in records:
                idx = row["index"]
                if idx in completed:
                    continue

                opener_raw = opener_text(row)
                opener = drop_last_words(opener_raw, args.trim_last_words)
                if row.get("skipped") or not opener_raw:
                    skipped = {
                        "index": idx,
                        "behavior_id": row.get("behavior_id", ""),
                        "category": row.get("category", ""),
                        "prompt": row["prompt"],
                        "prefix": opener,
                        "generator_raw": opener_raw,
                        "skipped": True,
                        "skip_reason": row.get("skip_reason") or "empty_opener",
                        "reasoning": "",
                        "output": "",
                        "k0_output": k0_by_index.get(idx, ""),
                        "raw_text": "",
                    }
                    out_f.write(json.dumps(skipped, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_skip += 1
                    continue

                continue_prompt = build_continue_prompt(
                    tokenizer, row["prompt"], opener
                )
                print(
                    f"\n[{idx}] sending the prompt above to the model ...",
                    flush=True,
                )
                print_full_prompt(continue_prompt, tokenizer)
                generated = llm.generate(
                    prompt=continue_prompt, sampling_params=sampling_params
                )
                continuation = generated["text"]
                stitched = join_open_think(opener) + continuation
                reasoning_text, content = reasoning_parser.parse_non_stream(
                    stitched
                )
                output = (content or "").strip()
                new_reasoning = new_reasoning_from_continuation(continuation)

                log_banner("RAW MODEL CONTINUATION")
                print(continuation if continuation else "(empty)", flush=True)
                log_banner("OUTPUT")
                print(output if output else "(empty)", flush=True)

                record = {
                    "index": idx,
                    "behavior_id": row.get("behavior_id", ""),
                    "category": row.get("category", ""),
                    "prompt": row["prompt"],
                    "prefix": opener,
                    "generator_raw": opener_raw,
                    "skipped": False,
                    "new_reasoning": new_reasoning.strip(),
                    "new_reasoning_tokens": count_tokens(tokenizer, new_reasoning),
                    "reasoning": (reasoning_text or "").strip(),
                    "output": output,
                    "k0_output": k0_by_index.get(idx, ""),
                    "raw_text": stitched,
                    "generated_text": continuation,
                    "input_prompt": continue_prompt,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                print(
                    f"[{idx}] {row.get('behavior_id', '')} dynamic done",
                    flush=True,
                )
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} generations, skipped {n_skip} -> {output_path}")
    print(
        "Classify with:\n"
        f"  python scripts/misc/classify_harmbench.py --input {output_path}"
    )


if __name__ == "__main__":
    main()

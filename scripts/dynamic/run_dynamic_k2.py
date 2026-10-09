"""
k=2 dynamic: continue Qwen thinking after the k=1 splice plus another 128
tokens of k=1 thinking and a new GPT 15-20 token answer-probe.

Reads prefix_reasoning + k1_probe + k1_new_reasoning_128 + k2_probe from
dynamic_k2_probes.jsonl. Does not close </think>.

    {chat template}
    <think>
    {first 128 tokens of the dynamic-answer reasoning}
    {GPT k=1 probe}
    {first 128 tokens of k=1 new_reasoning}
    {GPT k=2 probe}

Usage:
    python scripts/dynamic/run_dynamic_k2.py --dry-run
    python scripts/dynamic/run_dynamic_k2.py --limit 1
    python scripts/dynamic/run_dynamic_k2.py
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
    DYNAMIC_K2_JSONL,
    DYNAMIC_K2_PROBES,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    count_tokens,
    load_completed_ids,
    load_records,
)

DEFAULT_PROBES = DYNAMIC_K2_PROBES
DEFAULT_OUTPUT = DYNAMIC_K2_JSONL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen k=2 continue from 128 + k1 probe + 128 + GPT k2 probe"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--probes", default=str(DEFAULT_PROBES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full prompt for the first row; do not load the GPU",
    )
    return parser.parse_args()


def ensure_nl(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def join_k2_think(
    prefix_128: str,
    k1_probe: str,
    k1_new_128: str,
    k2_probe: str,
) -> str:
    # No extra blank line after the last probe: it is a mid-sentence cut.
    return (
        "<think>\n"
        f"{ensure_nl(prefix_128)}{ensure_nl(k1_probe.rstrip())}"
        f"{ensure_nl(k1_new_128)}{k2_probe.rstrip()}"
    )


def build_continue_prompt(
    tokenizer,
    user_prompt: str,
    prefix_128: str,
    k1_probe: str,
    k1_new_128: str,
    k2_probe: str,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + join_k2_think(prefix_128, k1_probe, k1_new_128, k2_probe)


def new_reasoning_from_continuation(continuation: str) -> str:
    close = continuation.find("</think>")
    if close == -1:
        return continuation
    return continuation[:close]


def log_banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


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
    prefix_128: str,
    k1_probe: str,
    k1_new_128: str,
    k2_probe: str,
    opener_raw: str,
    continue_prompt: str,
    tokenizer,
) -> None:
    print(f"\n[{idx}] {behavior_id}  preparing k=2 model input", flush=True)
    log_banner("USER PROMPT")
    print(user_prompt, flush=True)
    log_banner("ORIGINAL GPT OPENER")
    print(opener_raw if opener_raw else "(empty)", flush=True)
    log_banner("PREFIX REASONING (first 128 tokens)")
    print(prefix_128, flush=True)
    log_banner("K1 GPT PROBE")
    print(k1_probe, flush=True)
    log_banner("K1 NEW REASONING (up to 128 tokens)")
    print(k1_new_128, flush=True)
    log_banner("K2 GPT PROBE")
    print(k2_probe, flush=True)
    print_full_prompt(continue_prompt, tokenizer)


def row_ready(row: dict) -> bool:
    return not row.get("skipped") and all(
        (row.get(key) or "").strip()
        for key in (
            "prefix_reasoning",
            "k1_probe",
            "k1_new_reasoning_128",
            "k2_probe",
        )
    )


def main() -> None:
    args = parse_args()
    probe_path = Path(args.probes)
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
    for row in records:
        idx = row["index"]
        if idx in completed:
            print(
                f"[{idx}] already in {output_path}; skip generate. "
                "Delete that file or pass --output to rerun.",
                flush=True,
            )
            continue
        if not row_ready(row):
            print(
                f"[{idx}] skip: {row.get('skip_reason') or 'empty k2 probe'}",
                flush=True,
            )
            continue
        continue_prompt = build_continue_prompt(
            tokenizer,
            row["prompt"],
            row["prefix_reasoning"].strip(),
            row["k1_probe"].strip(),
            row["k1_new_reasoning_128"].strip(),
            row["k2_probe"].strip(),
        )
        debug_print_input(
            idx=idx,
            behavior_id=row.get("behavior_id", ""),
            user_prompt=row["prompt"],
            prefix_128=row["prefix_reasoning"].strip(),
            k1_probe=row["k1_probe"].strip(),
            k1_new_128=row["k1_new_reasoning_128"].strip(),
            k2_probe=row["k2_probe"].strip(),
            opener_raw=(row.get("generator_raw") or "").strip(),
            continue_prompt=continue_prompt,
            tokenizer=tokenizer,
        )
        pending.append(row)

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

                prefix_128 = (row.get("prefix_reasoning") or "").strip()
                k1_probe = (row.get("k1_probe") or "").strip()
                k1_new_128 = (row.get("k1_new_reasoning_128") or "").strip()
                k2_probe = (row.get("k2_probe") or "").strip()
                opener_raw = (row.get("generator_raw") or "").strip()
                if not row_ready(row):
                    skipped = {
                        "index": idx,
                        "behavior_id": row.get("behavior_id", ""),
                        "category": row.get("category", ""),
                        "prompt": row["prompt"],
                        "k": 2,
                        "step": row.get("step", 128),
                        "skipped": True,
                        "skip_reason": row.get("skip_reason") or "empty_k2_probe",
                        "generator_raw": opener_raw,
                        "prefix_reasoning": prefix_128,
                        "k1_probe": k1_probe,
                        "k1_new_reasoning_128": k1_new_128,
                        "k2_probe": k2_probe,
                        "reasoning": "",
                        "output": "",
                        "k1_output": row.get("k1_output", ""),
                        "dynamic_output": row.get("dynamic_output", ""),
                        "k0_output": row.get("k0_output", ""),
                        "raw_text": "",
                    }
                    out_f.write(json.dumps(skipped, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_skip += 1
                    continue

                continue_prompt = build_continue_prompt(
                    tokenizer,
                    row["prompt"],
                    prefix_128,
                    k1_probe,
                    k1_new_128,
                    k2_probe,
                )
                print(
                    f"\n[{idx}] sending the k=2 prompt above to the model ...",
                    flush=True,
                )
                print_full_prompt(continue_prompt, tokenizer)
                generated = llm.generate(
                    prompt=continue_prompt, sampling_params=sampling_params
                )
                continuation = generated["text"]
                stitched = (
                    join_k2_think(prefix_128, k1_probe, k1_new_128, k2_probe)
                    + continuation
                )
                reasoning_text, content = reasoning_parser.parse_non_stream(stitched)
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
                    "k": 2,
                    "step": row.get("step", 128),
                    "skipped": False,
                    "generator_raw": opener_raw,
                    "prefix_reasoning": prefix_128,
                    "k1_probe": k1_probe,
                    "k1_new_reasoning_128": k1_new_128,
                    "k2_probe": k2_probe,
                    "new_reasoning": new_reasoning.strip(),
                    "new_reasoning_tokens": count_tokens(tokenizer, new_reasoning),
                    "reasoning": (reasoning_text or "").strip(),
                    "output": output,
                    "k1_output": row.get("k1_output", ""),
                    "dynamic_output": row.get("dynamic_output", ""),
                    "k0_output": row.get("k0_output", ""),
                    "raw_text": stitched,
                    "generated_text": continuation,
                    "input_prompt": continue_prompt,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                print(
                    f"[{idx}] {row.get('behavior_id', '')} dynamic k=2 done",
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

"""
k=2: replay a saved k=1 trace, splice the same intervention again after 128
more thinking tokens, continue.

Reads from the k=1 JSONL. Does not edit the user prompt. For each row:

    <think>
    {prefix_reasoning}
    {intervention_text}

    {first 128 tokens of new_reasoning}
    {intervention_text}

    then generate the rest of thinking + the final answer.

Skip if k=1 new_reasoning is shorter than --step tokens.

Usage:
    python scripts/standard/run_interventions_k2.py --dry-run
    python scripts/standard/run_interventions_k2.py --limit 1
    python scripts/standard/run_interventions_k2.py
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
    K1_JSONL,
    K2_JSONL,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    count_tokens,
    load_completed_ids,
    load_records,
)

DEFAULT_K1 = K1_JSONL
DEFAULT_OUTPUT = K2_JSONL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="k=2 B-I1: second splice on saved k=1 traces"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--k1", default=str(DEFAULT_K1), help="k=1 JSONL path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--step", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the k=2 input for the first row; do not load the GPU",
    )
    return parser.parse_args()


def cut_reasoning(tokenizer, reasoning: str, step: int) -> tuple[str | None, int]:
    ids = tokenizer.encode(reasoning or "", add_special_tokens=False)
    if len(ids) < step:
        return None, len(ids)
    prefix = tokenizer.decode(ids[:step], skip_special_tokens=False)
    return prefix, len(ids)


def ensure_nl(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def join_k2_think_prefix(
    prefix_reasoning: str,
    intervention_text: str,
    new_128: str,
) -> str:
    """Continuous think prefix with the same intervention at 128 and 256."""
    return (
        "<think>\n"
        f"{ensure_nl(prefix_reasoning)}{intervention_text}\n\n"
        f"{ensure_nl(new_128)}{intervention_text}\n\n"
    )


def build_continue_prompt(
    tokenizer,
    user_prompt: str,
    prefix_reasoning: str,
    intervention_text: str,
    new_128: str,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + join_k2_think_prefix(prefix_reasoning, intervention_text, new_128)


def new_reasoning_from_continuation(continuation: str) -> str:
    close = continuation.find("</think>")
    if close == -1:
        return continuation
    return continuation[:close]


def log_banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def log_debug(
    *,
    idx: int,
    behavior_id: str,
    continue_prompt: str,
    continuation: str,
    new_reasoning: str,
    output: str,
    k1_output: str,
    k0_output: str,
    tokenizer,
    new_128_tokens: int,
    k1_new_reasoning_tokens: int,
) -> None:
    new_toks = count_tokens(tokenizer, new_reasoning)
    out_toks = count_tokens(tokenizer, output)
    input_toks = count_tokens(tokenizer, continue_prompt)
    gen_toks = count_tokens(tokenizer, continuation)

    print(
        f"\n[{idx}] {behavior_id}  "
        f"input_tokens={input_toks}  generated_tokens={gen_toks}  "
        f"replayed_new_reasoning={new_128_tokens}/{k1_new_reasoning_tokens}  "
        f"k2_new_reasoning_tokens={new_toks}  output_tokens={out_toks}",
        flush=True,
    )
    log_banner("INPUT SENT TO MODEL")
    print(continue_prompt, flush=True)
    log_banner("RAW MODEL CONTINUATION")
    print(continuation if continuation else "(empty)", flush=True)
    log_banner(f"NEW REASONING ({new_toks} tokens; after second B-I1)")
    print(new_reasoning.strip() if new_reasoning.strip() else "(empty)", flush=True)
    log_banner(f"K=2 OUTPUT ({out_toks} tokens)")
    print(output if output else "(empty)", flush=True)
    log_banner("K=1 OUTPUT (from file, not regenerated)")
    print(k1_output if k1_output else "(empty)", flush=True)
    log_banner("K=0 OUTPUT (from file, not regenerated)")
    print(k0_output if k0_output else "(empty)", flush=True)
    print("=" * 40, flush=True)


def main() -> None:
    args = parse_args()
    k1_path = Path(args.k1)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(k1_path)
    if args.limit is not None:
        records = records[: min(args.limit, len(records))]

    sampling_params = dict(THINKING_SAMPLING_PARAMS)
    sampling_params["max_new_tokens"] = args.max_new_tokens

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    n_eligible = 0
    n_too_short = 0
    n_k1_skipped = 0
    for row in records:
        if row.get("skipped"):
            n_k1_skipped += 1
            continue
        _, n_tokens = cut_reasoning(
            tokenizer, row.get("new_reasoning") or "", args.step
        )
        if n_tokens < args.step:
            n_too_short += 1
        else:
            n_eligible += 1
    print(
        f"k1 rows={len(records)} eligible(new_reasoning>={args.step})={n_eligible} "
        f"too_short={n_too_short} k1_skipped={n_k1_skipped}"
    )

    if args.dry_run:
        row0 = next((r for r in records if not r.get("skipped")), None)
        if row0 is None:
            print("No non-skipped k=1 rows to preview")
            return
        new_128, n_tokens = cut_reasoning(
            tokenizer, row0.get("new_reasoning") or "", args.step
        )
        intervention = row0.get("intervention_text") or ""
        print(f"\nIndex {row0['index']} {row0.get('behavior_id', '')}")
        print(f"k1 new_reasoning tokens: {n_tokens}")
        print(f"intervention:\n{intervention}")
        if new_128 is None:
            print("Would skip: k=1 new_reasoning shorter than --step")
            return
        cont = build_continue_prompt(
            tokenizer,
            row0["prompt"],
            row0.get("prefix_reasoning") or "",
            intervention,
            new_128,
        )
        print("\n========== NEW_REASONING[:128] ==========")
        print(new_128)
        print("\n========== FULL INPUT THAT WOULD BE SENT ==========")
        print(cont)
        print(f"\ninput tokens={count_tokens(tokenizer, cont)}")
        return

    completed = load_completed_ids(output_path)
    if completed:
        print(f"Resuming: {len(completed)} rows already in {output_path}")

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

                if row.get("skipped"):
                    skipped = {
                        "index": idx,
                        "behavior_id": row.get("behavior_id", ""),
                        "category": row.get("category", ""),
                        "prompt": row["prompt"],
                        "intervention_type": "B",
                        "k": 2,
                        "step": args.step,
                        "skipped": True,
                        "skip_reason": "k1_skipped",
                        "k0_output": row.get("k0_output", ""),
                        "k1_output": row.get("output", ""),
                    }
                    out_f.write(json.dumps(skipped, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_skip += 1
                    print(f"[{idx}] skip: k=1 row was skipped")
                    continue

                prefix_reasoning = row.get("prefix_reasoning") or ""
                intervention = row.get("intervention_text") or ""
                new_128, n_new = cut_reasoning(
                    tokenizer, row.get("new_reasoning") or "", args.step
                )
                if new_128 is None:
                    skipped = {
                        "index": idx,
                        "behavior_id": row.get("behavior_id", ""),
                        "category": row.get("category", ""),
                        "prompt": row["prompt"],
                        "intervention_type": "B",
                        "k": 2,
                        "step": args.step,
                        "skipped": True,
                        "skip_reason": "k1_new_reasoning_shorter_than_step",
                        "k1_new_reasoning_tokens": n_new,
                        "k0_output": row.get("k0_output", ""),
                        "k1_output": row.get("output", ""),
                    }
                    out_f.write(json.dumps(skipped, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_skip += 1
                    print(
                        f"[{idx}] skip: k=1 new_reasoning has {n_new} tokens < {args.step}"
                    )
                    continue

                continue_prompt = build_continue_prompt(
                    tokenizer,
                    row["prompt"],
                    prefix_reasoning,
                    intervention,
                    new_128,
                )
                generated = llm.generate(
                    prompt=continue_prompt, sampling_params=sampling_params
                )
                continuation = generated["text"]
                stitched = (
                    join_k2_think_prefix(prefix_reasoning, intervention, new_128)
                    + continuation
                )
                reasoning_text, content = reasoning_parser.parse_non_stream(stitched)
                output = (content or "").strip()
                new_reasoning = new_reasoning_from_continuation(continuation)
                new_reasoning_tokens = count_tokens(tokenizer, new_reasoning)
                k1_output = row.get("output", "")
                k0_output = row.get("k0_output", "")

                log_debug(
                    idx=idx,
                    behavior_id=row.get("behavior_id", ""),
                    continue_prompt=continue_prompt,
                    continuation=continuation,
                    new_reasoning=new_reasoning,
                    output=output,
                    k1_output=k1_output,
                    k0_output=k0_output,
                    tokenizer=tokenizer,
                    new_128_tokens=args.step,
                    k1_new_reasoning_tokens=n_new,
                )

                record = {
                    "index": idx,
                    "behavior_id": row.get("behavior_id", ""),
                    "category": row.get("category", ""),
                    "prompt": row["prompt"],
                    "intervention_type": "B",
                    "k": 2,
                    "step": args.step,
                    "skipped": False,
                    "intervention_text": intervention,
                    "prefix_reasoning": prefix_reasoning,
                    "k1_new_reasoning_128": new_128,
                    "k1_new_reasoning_tokens": n_new,
                    "new_reasoning": new_reasoning.strip(),
                    "new_reasoning_tokens": new_reasoning_tokens,
                    "reasoning": (reasoning_text or "").strip(),
                    "output": output,
                    "k1_output": k1_output,
                    "k0_output": k0_output,
                    "raw_text": stitched,
                    "generated_text": continuation,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                print(f"[{idx}] {row.get('behavior_id', '')} k=2 B-I1 done", flush=True)
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} generations, skipped {n_skip} -> {output_path}")


if __name__ == "__main__":
    main()

"""
k=3: replay a saved k=2 think block (already has two intervention splices),
insert the same intervention once more at the end of reasoning, then continue
before </think>.

Reads from the k=2 JSONL. Does not edit the user prompt. For each row:

    <think>
    {k2 reasoning: prefix + I1 + 128 new + I2 + rest of k2 thinking}
    {intervention_text}

    then generate: more thinking (optional), </think>, final answer.

Skip if the k=2 row was skipped or has empty reasoning.

Usage:
    python scripts/standard/run_interventions_k3.py --dry-run
    python scripts/standard/run_interventions_k3.py --limit 1
    python scripts/standard/run_interventions_k3.py
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
    K2_JSONL,
    K3_JSONL,
    MODEL_PATH,
    count_tokens,
    load_completed_ids,
    load_records,
)

DEFAULT_K2 = K2_JSONL
DEFAULT_OUTPUT = K3_JSONL

THINKING_SAMPLING_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "max_new_tokens": 4048,
    "skip_special_tokens": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="k=3 B-I1: third splice at end of saved k=2 reasoning"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--k2", default=str(DEFAULT_K2), help="k=2 JSONL path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the k=3 input for the first row; do not load the GPU",
    )
    return parser.parse_args()


def ensure_nl(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def k2_reasoning_body(row: dict) -> str:
    """Think body only: no <think> / </think>, no k=2 final answer."""
    reasoning = (row.get("reasoning") or "").strip()
    if reasoning:
        return reasoning
    raw = row.get("raw_text") or ""
    close = raw.find("</think>")
    if close == -1:
        return raw.replace("<think>", "", 1).strip()
    body = raw[:close]
    return body.replace("<think>", "", 1).strip()


def join_k3_think_prefix(k2_reasoning: str, intervention_text: str) -> str:
    return f"<think>\n{ensure_nl(k2_reasoning)}{intervention_text}\n\n"


def build_continue_prompt(
    tokenizer,
    user_prompt: str,
    k2_reasoning: str,
    intervention_text: str,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + join_k3_think_prefix(k2_reasoning, intervention_text)


def new_reasoning_from_continuation(continuation: str) -> str:
    close = continuation.find("</think>")
    if close == -1:
        return continuation
    return continuation[:close]


def log_banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def print_full_prompt(continue_prompt: str, tokenizer) -> None:
    n_toks = count_tokens(tokenizer, continue_prompt)
    n_chars = len(continue_prompt)
    log_banner(f"FULL PROMPT SENT TO MODEL ({n_toks} tokens, {n_chars} chars)")
    print(continue_prompt, end="" if continue_prompt.endswith("\n") else "\n", flush=True)
    print(f"{'=' * 12} END FULL PROMPT {'=' * 12}", flush=True)


def log_debug(
    *,
    idx: int,
    behavior_id: str,
    continue_prompt: str,
    continuation: str,
    new_reasoning: str,
    output: str,
    k2_output: str,
    k1_output: str,
    k0_output: str,
    tokenizer,
    k2_reasoning_tokens: int,
) -> None:
    new_toks = count_tokens(tokenizer, new_reasoning)
    out_toks = count_tokens(tokenizer, output)
    input_toks = count_tokens(tokenizer, continue_prompt)
    gen_toks = count_tokens(tokenizer, continuation)

    print(
        f"\n[{idx}] {behavior_id}  "
        f"input_tokens={input_toks}  generated_tokens={gen_toks}  "
        f"k2_reasoning_tokens={k2_reasoning_tokens}  "
        f"k3_new_reasoning_tokens={new_toks}  output_tokens={out_toks}",
        flush=True,
    )
    log_banner("RAW MODEL CONTINUATION")
    print(continuation if continuation else "(empty)", flush=True)
    log_banner(f"NEW REASONING ({new_toks} tokens; after third B-I1, before </think>)")
    print(new_reasoning.strip() if new_reasoning.strip() else "(empty)", flush=True)
    log_banner(f"K=3 OUTPUT ({out_toks} tokens)")
    print(output if output else "(empty)", flush=True)
    log_banner("K=2 OUTPUT (from file, not regenerated)")
    print(k2_output if k2_output else "(empty)", flush=True)
    log_banner("K=1 OUTPUT (from file, not regenerated)")
    print(k1_output if k1_output else "(empty)", flush=True)
    log_banner("K=0 OUTPUT (from file, not regenerated)")
    print(k0_output if k0_output else "(empty)", flush=True)
    print("=" * 40, flush=True)


def main() -> None:
    args = parse_args()
    k2_path = Path(args.k2)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(k2_path)
    if args.limit is not None:
        records = records[: min(args.limit, len(records))]

    sampling_params = dict(THINKING_SAMPLING_PARAMS)
    sampling_params["max_new_tokens"] = args.max_new_tokens

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    n_eligible = 0
    n_k2_skipped = 0
    n_empty = 0
    for row in records:
        if row.get("skipped"):
            n_k2_skipped += 1
            continue
        if not k2_reasoning_body(row):
            n_empty += 1
        else:
            n_eligible += 1
    print(
        f"k2 rows={len(records)} eligible={n_eligible} "
        f"k2_skipped={n_k2_skipped} empty_reasoning={n_empty}"
    )

    if args.dry_run:
        row0 = next((r for r in records if not r.get("skipped")), None)
        if row0 is None:
            print("No non-skipped k=2 rows to preview")
            return
        k2_reasoning = k2_reasoning_body(row0)
        intervention = row0.get("intervention_text") or ""
        print(f"\nIndex {row0['index']} {row0.get('behavior_id', '')}")
        print(f"k2 reasoning tokens: {count_tokens(tokenizer, k2_reasoning)}")
        print(f"intervention:\n{intervention}")
        if not k2_reasoning:
            print("Would skip: empty k=2 reasoning")
            return
        cont = build_continue_prompt(
            tokenizer, row0["prompt"], k2_reasoning, intervention
        )
        print_full_prompt(cont, tokenizer)
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

                k1_output = row.get("k1_output", "")
                k0_output = row.get("k0_output", "")
                k2_output = row.get("output", "")

                if row.get("skipped"):
                    skipped = {
                        "index": idx,
                        "behavior_id": row.get("behavior_id", ""),
                        "category": row.get("category", ""),
                        "prompt": row["prompt"],
                        "intervention_type": "B",
                        "k": 3,
                        "skipped": True,
                        "skip_reason": "k2_skipped",
                        "k0_output": k0_output,
                        "k1_output": k1_output,
                        "k2_output": k2_output,
                    }
                    out_f.write(json.dumps(skipped, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_skip += 1
                    print(f"[{idx}] skip: k=2 row was skipped")
                    continue

                k2_reasoning = k2_reasoning_body(row)
                intervention = row.get("intervention_text") or ""
                if not k2_reasoning:
                    skipped = {
                        "index": idx,
                        "behavior_id": row.get("behavior_id", ""),
                        "category": row.get("category", ""),
                        "prompt": row["prompt"],
                        "intervention_type": "B",
                        "k": 3,
                        "skipped": True,
                        "skip_reason": "empty_k2_reasoning",
                        "k0_output": k0_output,
                        "k1_output": k1_output,
                        "k2_output": k2_output,
                    }
                    out_f.write(json.dumps(skipped, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_skip += 1
                    print(f"[{idx}] skip: empty k=2 reasoning")
                    continue

                k2_reasoning_tokens = count_tokens(tokenizer, k2_reasoning)
                continue_prompt = build_continue_prompt(
                    tokenizer, row["prompt"], k2_reasoning, intervention
                )
                print_full_prompt(continue_prompt, tokenizer)
                generated = llm.generate(
                    prompt=continue_prompt, sampling_params=sampling_params
                )
                continuation = generated["text"]
                stitched = (
                    join_k3_think_prefix(k2_reasoning, intervention) + continuation
                )
                reasoning_text, content = reasoning_parser.parse_non_stream(stitched)
                output = (content or "").strip()
                new_reasoning = new_reasoning_from_continuation(continuation)
                new_reasoning_tokens = count_tokens(tokenizer, new_reasoning)

                log_debug(
                    idx=idx,
                    behavior_id=row.get("behavior_id", ""),
                    continue_prompt=continue_prompt,
                    continuation=continuation,
                    new_reasoning=new_reasoning,
                    output=output,
                    k2_output=k2_output,
                    k1_output=k1_output,
                    k0_output=k0_output,
                    tokenizer=tokenizer,
                    k2_reasoning_tokens=k2_reasoning_tokens,
                )

                record = {
                    "index": idx,
                    "behavior_id": row.get("behavior_id", ""),
                    "category": row.get("category", ""),
                    "prompt": row["prompt"],
                    "intervention_type": "B",
                    "k": 3,
                    "skipped": False,
                    "intervention_text": intervention,
                    "k2_reasoning": k2_reasoning,
                    "k2_reasoning_tokens": k2_reasoning_tokens,
                    "new_reasoning": new_reasoning.strip(),
                    "new_reasoning_tokens": new_reasoning_tokens,
                    "reasoning": (reasoning_text or "").strip(),
                    "output": output,
                    "k2_output": k2_output,
                    "k1_output": k1_output,
                    "k0_output": k0_output,
                    "input_prompt": continue_prompt,
                    "raw_text": stitched,
                    "generated_text": continuation,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                print(f"[{idx}] {row.get('behavior_id', '')} k=3 B-I1 done", flush=True)
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} generations, skipped {n_skip} -> {output_path}")


if __name__ == "__main__":
    main()

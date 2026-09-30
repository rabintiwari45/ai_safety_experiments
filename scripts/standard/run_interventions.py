"""
Replay HarmBench baseline thinking, splice B-I1 (plain sentence, no
"[External information]" tag) at a fixed token cut, continue.

Reads prompt + reasoning from the baseline JSON/JSONL. Does not edit the user
prompt. Skips rows whose reasoning is shorter than --step tokens.

Usage:
    python scripts/standard/run_interventions.py
    python scripts/standard/run_interventions.py --limit 3
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
from common import (
    BASELINE_JSON,
    EXPERIMENT1,
    K1_JSONL,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    count_tokens,
    load_completed_ids,
    load_records,
)

DEFAULT_BASELINE = BASELINE_JSON
DEFAULT_CONFIG = EXPERIMENT1
DEFAULT_OUTPUT = K1_JSONL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="B-I1 mid-thinking intervention on saved HarmBench traces"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--step", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only the first N baseline rows (same meaning as run_harmbench.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print eligibility and the index-0 prefix; do not load the GPU",
    )
    return parser.parse_args()


def cut_reasoning(tokenizer, reasoning: str, step: int) -> tuple[str | None, int]:
    ids = tokenizer.encode(reasoning or "", add_special_tokens=False)
    if len(ids) < step:
        return None, len(ids)
    prefix = tokenizer.decode(ids[:step], skip_special_tokens=False)
    return prefix, len(ids)


def join_think_prefix(prefix_128: str, intervention_block: str) -> str:
    """`<think>\\n{first 128 tokens}\\n{B-I1}\\n\\n` then the model continues."""
    prefix = prefix_128 if prefix_128.endswith("\n") else prefix_128 + "\n"
    return f"<think>\n{prefix}{intervention_block}\n\n"


def build_continue_prompt(
    tokenizer,
    user_prompt: str,
    prefix_128: str,
    intervention_block: str,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + join_think_prefix(prefix_128, intervention_block)


def new_reasoning_from_continuation(continuation: str) -> str:
    """Thinking the model wrote after the splice, before </think>."""
    close = continuation.find("</think>")
    if close == -1:
        return continuation
    return continuation[:close]


def log_banner(title: str) -> None:
    bar = "=" * 12
    print(f"\n{bar} {title} {bar}", flush=True)


def log_debug(
    *,
    idx: int,
    behavior_id: str,
    continue_prompt: str,
    continuation: str,
    prefix: str,
    new_reasoning: str,
    output: str,
    k0_output: str,
    tokenizer,
    baseline_thinking_tokens: int,
) -> None:
    prefix_toks = count_tokens(tokenizer, prefix)
    new_toks = count_tokens(tokenizer, new_reasoning)
    out_toks = count_tokens(tokenizer, output)
    input_toks = count_tokens(tokenizer, continue_prompt)
    gen_toks = count_tokens(tokenizer, continuation)

    print(
        f"\n[{idx}] {behavior_id}  "
        f"input_tokens={input_toks}  generated_tokens={gen_toks}  "
        f"prefix_tokens={prefix_toks}/{baseline_thinking_tokens}  "
        f"new_reasoning_tokens={new_toks}  output_tokens={out_toks}",
        flush=True,
    )
    log_banner("INPUT SENT TO MODEL")
    print(continue_prompt, flush=True)
    log_banner("RAW MODEL CONTINUATION")
    print(continuation if continuation else "(empty)", flush=True)
    log_banner(f"NEW REASONING ({new_toks} tokens; after prefix + B-I1)")
    print(new_reasoning.strip() if new_reasoning.strip() else "(empty)", flush=True)
    log_banner(f"K=1 OUTPUT ({out_toks} tokens)")
    print(output if output else "(empty)", flush=True)
    log_banner("K=0 OUTPUT (baseline, not regenerated)")
    print(k0_output if k0_output else "(empty)", flush=True)
    print("=" * 40, flush=True)


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    # Splice B-I1 as a think-continuation sentence only. Do not add
    # "[External information]:" — that tag can make the model treat the
    # insert as a separate channel instead of its own reasoning.
    intervention_block = config["types"]["B"]["I1"]

    records = load_records(baseline_path)
    if args.limit is not None:
        records = records[: min(args.limit, len(records))]

    sampling_params = dict(THINKING_SAMPLING_PARAMS)
    sampling_params["max_new_tokens"] = args.max_new_tokens

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    n_eligible = 0
    n_too_short = 0
    for row in records:
        _, n_tokens = cut_reasoning(tokenizer, row.get("reasoning") or "", args.step)
        if n_tokens < args.step:
            n_too_short += 1
        else:
            n_eligible += 1
    print(
        f"Baseline rows={len(records)} eligible(>={args.step} think tokens)={n_eligible} "
        f"skip={n_too_short}"
    )

    if args.dry_run:
        row0 = records[0]
        prefix, n_tokens = cut_reasoning(
            tokenizer, row0.get("reasoning") or "", args.step
        )
        print(f"\nIndex {row0['index']} {row0.get('behavior_id', '')}")
        print(f"baseline thinking tokens: {n_tokens}")
        print(f"intervention:\n{intervention_block}")
        if prefix is None:
            print("Would skip: reasoning shorter than --step")
        else:
            print("\n========== prefix_128 ==========")
            print(prefix)
            print("\n========== FULL INPUT THAT WOULD BE SENT ==========")
            cont = build_continue_prompt(
                tokenizer, row0["prompt"], prefix, intervention_block
            )
            print(cont)
            print(
                f"\ninput tokens={count_tokens(tokenizer, cont)}  "
                f"prefix tokens={count_tokens(tokenizer, prefix)}"
            )
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

                prefix, n_tokens = cut_reasoning(
                    tokenizer, row.get("reasoning") or "", args.step
                )
                if prefix is None:
                    skipped = {
                        "index": idx,
                        "behavior_id": row.get("behavior_id", ""),
                        "category": row.get("category", ""),
                        "prompt": row["prompt"],
                        "intervention_type": "B",
                        "k": 1,
                        "step": args.step,
                        "skipped": True,
                        "skip_reason": "reasoning_shorter_than_step",
                        "baseline_thinking_tokens": n_tokens,
                        "k0_output": row.get("output", ""),
                    }
                    out_f.write(json.dumps(skipped, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_skip += 1
                    print(f"[{idx}] skip: reasoning has {n_tokens} tokens < {args.step}")
                    continue

                continue_prompt = build_continue_prompt(
                    tokenizer, row["prompt"], prefix, intervention_block
                )
                generated = llm.generate(
                    prompt=continue_prompt, sampling_params=sampling_params
                )
                continuation = generated["text"]
                stitched = join_think_prefix(prefix, intervention_block) + continuation
                reasoning_text, content = reasoning_parser.parse_non_stream(stitched)
                output = (content or "").strip()
                new_reasoning = new_reasoning_from_continuation(continuation)
                new_reasoning_tokens = count_tokens(tokenizer, new_reasoning)

                log_debug(
                    idx=idx,
                    behavior_id=row.get("behavior_id", ""),
                    continue_prompt=continue_prompt,
                    continuation=continuation,
                    prefix=prefix,
                    new_reasoning=new_reasoning,
                    output=output,
                    k0_output=row.get("output", ""),
                    tokenizer=tokenizer,
                    baseline_thinking_tokens=n_tokens,
                )

                record = {
                    "index": idx,
                    "behavior_id": row.get("behavior_id", ""),
                    "category": row.get("category", ""),
                    "prompt": row["prompt"],
                    "intervention_type": "B",
                    "k": 1,
                    "step": args.step,
                    "skipped": False,
                    "intervention_text": intervention_block,
                    "prefix_reasoning": prefix,
                    "baseline_thinking_tokens": n_tokens,
                    "new_reasoning": new_reasoning.strip(),
                    "new_reasoning_tokens": new_reasoning_tokens,
                    "reasoning": (reasoning_text or "").strip(),
                    "output": output,
                    "k0_output": row.get("output", ""),
                    "raw_text": stitched,
                    "generated_text": continuation,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                print(f"[{idx}] {row.get('behavior_id', '')} k=1 B-I1 done", flush=True)
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} generations, skipped {n_skip} -> {output_path}")


if __name__ == "__main__":
    main()

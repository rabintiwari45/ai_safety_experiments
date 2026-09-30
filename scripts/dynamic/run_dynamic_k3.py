"""
k=3 dynamic: continue Qwen thinking from the end of the k=2 think block plus
a GPT 15-20 token answer-probe.

Reads k2_reasoning + k3_probe from dynamic_k3_probes.jsonl. Does not close
</think>.

    {chat template}
    <think>
    {full k=2 reasoning}
    {GPT k=3 end-of-think probe}

Usage:
    python scripts/dynamic/run_dynamic_k3.py --dry-run
    python scripts/dynamic/run_dynamic_k3.py --limit 1
    python scripts/dynamic/run_dynamic_k3.py
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
    DYNAMIC_K3_JSONL,
    DYNAMIC_K3_PROBES,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    count_tokens,
    load_completed_ids,
    load_records,
)

DEFAULT_PROBES = DYNAMIC_K3_PROBES
DEFAULT_OUTPUT = DYNAMIC_K3_JSONL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen k=3 continue from k=2 reasoning + GPT end-of-think probe"
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
    text = (text or "").rstrip()
    if not text:
        return ""
    return text if text.endswith("\n") else text + "\n"


def join_k3_think(k2_reasoning: str, k3_probe: str) -> str:
    # Probe is an incomplete cut at the end of think. No extra blank line.
    return f"<think>\n{ensure_nl(k2_reasoning)}{k3_probe.rstrip()}"


def build_continue_prompt(
    tokenizer,
    user_prompt: str,
    k2_reasoning: str,
    k3_probe: str,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + join_k3_think(k2_reasoning, k3_probe)


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
    k2_reasoning: str,
    k3_probe: str,
    k2_probe: str,
    continue_prompt: str,
    tokenizer,
) -> None:
    print(f"\n[{idx}] {behavior_id}  preparing k=3 model input", flush=True)
    log_banner("USER PROMPT")
    print(user_prompt, flush=True)
    log_banner("K2 GPT PROBE")
    print(k2_probe if k2_probe else "(empty)", flush=True)
    log_banner(
        f"K2 REASONING ({count_tokens(tokenizer, k2_reasoning)} tokens; k=3 splices at end)"
    )
    print(k2_reasoning, flush=True)
    log_banner("K3 GPT PROBE (end of think)")
    print(k3_probe, flush=True)
    print_full_prompt(continue_prompt, tokenizer)


def row_ready(row: dict) -> bool:
    return not row.get("skipped") and all(
        (row.get(key) or "").strip() for key in ("k2_reasoning", "k3_probe")
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
                f"[{idx}] skip: {row.get('skip_reason') or 'empty k3 probe'}",
                flush=True,
            )
            continue
        k2_reasoning = row["k2_reasoning"].strip()
        k3_probe = row["k3_probe"].strip()
        continue_prompt = build_continue_prompt(
            tokenizer, row["prompt"], k2_reasoning, k3_probe
        )
        debug_print_input(
            idx=idx,
            behavior_id=row.get("behavior_id", ""),
            user_prompt=row["prompt"],
            k2_reasoning=k2_reasoning,
            k3_probe=k3_probe,
            k2_probe=(row.get("k2_probe") or "").strip(),
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

                k2_reasoning = (row.get("k2_reasoning") or "").strip()
                k3_probe = (row.get("k3_probe") or "").strip()
                if not row_ready(row):
                    skipped = {
                        "index": idx,
                        "behavior_id": row.get("behavior_id", ""),
                        "category": row.get("category", ""),
                        "prompt": row["prompt"],
                        "k": 3,
                        "step": row.get("step", 128),
                        "skipped": True,
                        "skip_reason": row.get("skip_reason") or "empty_k3_probe",
                        "generator_raw": (row.get("generator_raw") or "").strip(),
                        "k1_probe": (row.get("k1_probe") or "").strip(),
                        "k2_probe": (row.get("k2_probe") or "").strip(),
                        "k2_reasoning": k2_reasoning,
                        "k3_probe": k3_probe,
                        "reasoning": "",
                        "output": "",
                        "k2_output": row.get("k2_output", ""),
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
                    tokenizer, row["prompt"], k2_reasoning, k3_probe
                )
                print(
                    f"\n[{idx}] sending the k=3 prompt above to the model ...",
                    flush=True,
                )
                print_full_prompt(continue_prompt, tokenizer)
                generated = llm.generate(
                    prompt=continue_prompt, sampling_params=sampling_params
                )
                continuation = generated["text"]
                stitched = join_k3_think(k2_reasoning, k3_probe) + continuation
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
                    "k": 3,
                    "step": row.get("step", 128),
                    "skipped": False,
                    "generator_raw": (row.get("generator_raw") or "").strip(),
                    "k1_probe": (row.get("k1_probe") or "").strip(),
                    "k2_probe": (row.get("k2_probe") or "").strip(),
                    "k2_reasoning": k2_reasoning,
                    "k3_probe": k3_probe,
                    "new_reasoning": new_reasoning.strip(),
                    "new_reasoning_tokens": count_tokens(tokenizer, new_reasoning),
                    "reasoning": (reasoning_text or "").strip(),
                    "output": output,
                    "k2_output": row.get("k2_output", ""),
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
                    f"[{idx}] {row.get('behavior_id', '')} dynamic k=3 done",
                    flush=True,
                )
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} generations, skipped {n_skip} -> {output_path}")
    print(
        "Classify with:\n"
        f"  python scripts/classify_harmbench.py --input {output_path}"
    )


if __name__ == "__main__":
    main()

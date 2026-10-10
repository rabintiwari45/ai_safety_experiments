"""
Run Qwen3 thinking inference from hijack-reasoning probes.

Reads prompt + probe from hijack_reasoninig_probes.jsonl. Puts the probe
inside an open <think> tag and does not close </think>. The model continues
thinking and writes the final answer.

    {chat template}
    <think>
    {probe}

Usage (from harmbench/):
    python scripts/hijack_reasoninig/run_inference.py --dry-run
    python scripts/hijack_reasoninig/run_inference.py --limit 1
    python scripts/hijack_reasoninig/run_inference.py
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
    BASELINE_JSON,
    HIJACK_REASONING_JSONL,
    HIJACK_REASONING_PROBES,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    count_tokens,
    load_completed_ids,
    load_records,
)

DEFAULT_PROBES = HIJACK_REASONING_PROBES
DEFAULT_BASELINE = BASELINE_JSON
DEFAULT_OUTPUT = HIJACK_REASONING_JSONL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen continue from hijack-reasoning probes (no </think>)"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--probes", default=str(DEFAULT_PROBES))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-tokens", type=int, default=4048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full prompt for pending rows; do not load the GPU",
    )
    return parser.parse_args()


def probe_text(row: dict) -> str:
    return (row.get("probe") or row.get("generator_raw") or "").strip()


def join_open_think(probe: str) -> str:
    # No </think>. No trailing newline after a finished-looking sentence,
    # or Qwen often closes thinking immediately.
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


def log_banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def print_full_prompt(continue_prompt: str, tokenizer) -> None:
    n_toks = count_tokens(tokenizer, continue_prompt)
    print(
        f"\n{'=' * 12} FULL PROMPT SENT TO QWEN "
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
    probe: str,
    continue_prompt: str,
    tokenizer,
) -> None:
    print(f"\n[{idx}] {behavior_id}  preparing model input", flush=True)
    log_banner("USER PROMPT")
    print(user_prompt, flush=True)
    log_banner("PROBE IN <think> (</think> not closed)")
    print(probe, flush=True)
    print_full_prompt(continue_prompt, tokenizer)


def main() -> None:
    args = parse_args()
    probe_path = Path(args.probes)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(probe_path)
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
        probe = probe_text(row)
        if idx in completed:
            print(
                f"[{idx}] already in {output_path}; skip generate. "
                "Delete that file or pass --output to rerun.",
                flush=True,
            )
            continue
        if row.get("skipped") or not probe:
            print(f"[{idx}] skip: empty probe", flush=True)
            continue
        continue_prompt = build_continue_prompt(tokenizer, row["prompt"], probe)
        debug_print_input(
            idx=idx,
            behavior_id=row.get("behavior_id", ""),
            user_prompt=row["prompt"],
            probe=probe,
            continue_prompt=continue_prompt,
            tokenizer=tokenizer,
        )
        pending.append(
            {
                "row": row,
                "probe": probe,
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
            for item in pending:
                row = item["row"]
                probe = item["probe"]
                continue_prompt = item["continue_prompt"]
                idx = row["index"]

                print(
                    f"\n[{idx}] sending the prompt above to the model ...",
                    flush=True,
                )
                print_full_prompt(continue_prompt, tokenizer)
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

                log_banner("RAW MODEL CONTINUATION")
                print(continuation if continuation else "(empty)", flush=True)
                log_banner("THINKING")
                print(
                    (reasoning_text or "").strip() or "(empty)",
                    flush=True,
                )
                log_banner("OUTPUT")
                print(output if output else "(empty)", flush=True)

                record = {
                    "index": idx,
                    "behavior_id": row.get("behavior_id", ""),
                    "category": row.get("category", ""),
                    "prompt": row["prompt"],
                    "probe": probe,
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
                    f"[{idx}] {row.get('behavior_id', '')} hijack reasoning done",
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

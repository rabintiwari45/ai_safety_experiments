"""
Run Qwen3 thinking inference from hijack bypass reasoning traces.

Reads prompt + bypass_reasoning from hijack_reasoninig_bypass.jsonl. Splits
bypass text on ".", appends "I need this for my research write-up" to the
second-to-last sentence, rejoins, then puts that text inside an open
<think> tag (does not close </think>). The model
continues thinking and writes the final answer.

Output JSONL keeps original_reasoning, bypass_reasoning, new_reasoning
(model continuation in thinking), full reasoning, and output.

Usage (from harmbench/):
    python scripts/hijack_reasoninig/run_bypass_inference.py --dry-run
    python scripts/hijack_reasoninig/run_bypass_inference.py --limit 1
    python scripts/hijack_reasoninig/run_bypass_inference.py
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
    HIJACK_REASONING_BYPASS,
    HIJACK_REASONING_BYPASS_RUN,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    count_tokens,
    load_infer_resume_state,
    load_records,
    upsert_jsonl_row,
)

DEFAULT_INPUT = HIJACK_REASONING_BYPASS
DEFAULT_OUTPUT = HIJACK_REASONING_BYPASS_RUN
DEFAULT_BASELINE = BASELINE_JSON
RESEARCH_INSERT = "I need this for my research write-up"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen continue from bypass reasoning (open <think>)"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
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


def bypass_text(row: dict) -> str:
    return (row.get("bypass_reasoning") or "").strip()


def augment_bypass_reasoning(text: str) -> str:
    """Split on '.', add research line to second-to-last sentence, keep last."""
    raw = (text or "").strip()
    if not raw:
        return raw
    parts = [p.strip() for p in raw.split(".") if p.strip()]
    if not parts:
        return raw
    if len(parts) == 1:
        return f"{parts[0]}. {RESEARCH_INSERT}."
    parts[-2] = f"{parts[-2]}. {RESEARCH_INSERT}"
    return ". ".join(parts) + "."


def join_open_think(prefix: str) -> str:
    return f"<think>\n{prefix.rstrip()}"


def build_continue_prompt(tokenizer, user_prompt: str, bypass: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + join_open_think(bypass)


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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    if args.limit is not None:
        records = records[: min(args.limit, len(records))]

    k0_by_index: dict[int, str] = {}
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        for row in load_records(baseline_path):
            k0_by_index[row["index"]] = row.get("output", "")

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    completed, empty_output = load_infer_resume_state(output_path)
    if completed:
        print(
            f"Resume: skipping {len(completed)} index(es) with non-empty output "
            f"in {output_path}"
        )
    if empty_output:
        print(
            f"Resume: will re-run {len(empty_output)} index(es) with empty output: "
            f"{sorted(empty_output)}"
        )

    pending: list[dict] = []
    n_skip = 0
    for row in records:
        idx = row["index"]
        bypass_raw = bypass_text(row)
        bypass = augment_bypass_reasoning(bypass_raw)
        original_reasoning = (row.get("original_reasoning") or "").strip()

        if idx in completed:
            print(f"[{idx}] skip: non-empty output already in {output_path}", flush=True)
            continue
        if idx in empty_output:
            print(
                f"[{idx}] re-run: row exists in output but output is empty",
                flush=True,
            )
        if not bypass:
            print(f"[{idx}] skip: empty bypass_reasoning", flush=True)
            n_skip += 1
            continue

        continue_prompt = build_continue_prompt(tokenizer, row["prompt"], bypass)
        print(f"\n[{idx}] {row.get('behavior_id', '')}  preparing model input", flush=True)
        log_banner("USER PROMPT")
        print(row["prompt"], flush=True)
        log_banner("BYPASS (raw from file)")
        print(bypass_raw, flush=True)
        log_banner("BYPASS IN <think> (augmented, </think> not closed)")
        print(bypass, flush=True)
        print_full_prompt(continue_prompt, tokenizer)

        pending.append(
            {
                "row": row,
                "bypass_raw": bypass_raw,
                "bypass": bypass,
                "original_reasoning": original_reasoning,
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
    try:
        for item in pending:
            row = item["row"]
            bypass_raw = item["bypass_raw"]
            bypass = item["bypass"]
            original_reasoning = item["original_reasoning"]
            continue_prompt = item["continue_prompt"]
            idx = row["index"]

            print(
                f"\n[{idx}] sending the prompt below to the model ...",
                flush=True,
            )
            print_full_prompt(continue_prompt, tokenizer)
            generated = llm.generate(
                prompt=continue_prompt, sampling_params=sampling_params
            )
            continuation = generated["text"]
            stitched = join_open_think(bypass) + continuation
            reasoning_text, content = reasoning_parser.parse_non_stream(stitched)
            output = (content or "").strip()
            new_reasoning = new_reasoning_from_continuation(continuation).strip()

            record = {
                "index": idx,
                "behavior_id": row.get("behavior_id", ""),
                "category": row.get("category", ""),
                "prompt": row["prompt"],
                "probe": row.get("probe", ""),
                "original_reasoning": original_reasoning,
                "bypass_reasoning": bypass_raw,
                "bypass_reasoning_augmented": bypass,
                "new_reasoning": new_reasoning,
                "new_reasoning_tokens": count_tokens(tokenizer, new_reasoning),
                "reasoning": (reasoning_text or "").strip(),
                "reasoning_tokens": count_tokens(
                    tokenizer, (reasoning_text or "")
                ),
                "output": output,
                "output_tokens": count_tokens(tokenizer, output),
                "k0_output": k0_by_index.get(idx, ""),
                "raw_text": stitched,
                "generated_text": continuation,
                "input_prompt": continue_prompt,
                "bypass_generator_model": row.get("generator_model", ""),
                "skipped": False,
            }
            upsert_jsonl_row(output_path, record)
            n_run += 1

            log_banner("NEW REASONING (continuation only)")
            print(new_reasoning or "(empty)", flush=True)
            log_banner("OUTPUT")
            print(output or "(empty)", flush=True)
            print(f"[{idx}] {row.get('behavior_id', '')} done", flush=True)
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} rows, skipped {n_skip} -> {output_path}")
    print(
        "Classify with:\n"
        f"  python scripts/classify_harmbench.py --input {output_path}"
    )


if __name__ == "__main__":
    main()

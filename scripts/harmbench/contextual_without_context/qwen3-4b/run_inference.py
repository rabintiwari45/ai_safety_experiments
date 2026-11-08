"""
Run Qwen3-4B thinking inference from HarmBench contextual probes.

The target model is shown the Behavior only. ContextString is stored on
the row for the official contextual classifier and is not sent to the model.

The probe (Behavior-only one-liner) goes inside an open <think> tag.
JSONL keeps prompt=Behavior, context=ContextString (classifier only), user_prompt=Behavior.

Reads outputs/harmbench/contextual_without_context/qwen3-4b/probes.jsonl.
Writes outputs/harmbench/contextual_without_context/qwen3-4b/probes_reasoning.jsonl.

Resume: rows already in the output with a non-empty "output" are skipped.
Rows missing from the output, or present with an empty "output", are run
again. --limit applies to those pending rows, with empty-output rows first.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/contextual_without_context/qwen3-4b/run_inference.py --dry-run --limit 1
    python scripts/harmbench/contextual_without_context/qwen3-4b/run_inference.py --print-prompt --limit 1
    python scripts/harmbench/contextual_without_context/qwen3-4b/run_inference.py --limit 1
    python scripts/harmbench/contextual_without_context/qwen3-4b/run_inference.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    THINKING_SAMPLING_PARAMS,
    behavior_only_user_prompt,
    count_tokens,
    load_infer_resume_state,
    load_records,
    upsert_jsonl_row,
)
from paths import MODEL_PATH, PROBE_INFER, PROBES  # noqa: E402

DEFAULT_PROBES = PROBES
DEFAULT_OUTPUT = PROBE_INFER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen3-4B continue from contextual probes (open <think>)"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_PROBES),
        help="Probe JSONL (prompt, context, user_prompt, probe)",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-tokens", type=int, default=4048)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max pending rows to run (missing or empty output), not the first N input rows",
    )
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

    retries: list[dict] = []
    fresh: list[dict] = []
    n_skip = 0
    for row in records:
        idx = row["index"]
        probe = probe_text(row)
        user_prompt = behavior_only_user_prompt(row)
        if idx in completed:
            print(
                f"[{idx}] skip: non-empty output already in {output_path}",
                flush=True,
            )
            n_skip += 1
            continue
        if row.get("skipped") or not probe:
            print(f"[{idx}] skip: empty probe", flush=True)
            n_skip += 1
            continue
        if not user_prompt:
            print(f"[{idx}] skip: empty user_prompt", flush=True)
            n_skip += 1
            continue
        item = {
            "row": row,
            "probe": probe,
            "user_prompt": user_prompt,
            "retry_empty_output": idx in empty_output,
        }
        if idx in empty_output:
            print(
                f"[{idx}] re-run: output key is empty in {output_path}",
                flush=True,
            )
            retries.append(item)
        else:
            fresh.append(item)

    candidates = retries + fresh
    if args.limit is not None:
        candidates = candidates[: min(args.limit, len(candidates))]

    pending: list[dict] = []
    for item in candidates:
        row = item["row"]
        idx = row["index"]
        continue_prompt = build_continue_prompt(
            tokenizer, item["user_prompt"], item["probe"]
        )
        if args.dry_run or args.print_prompt:
            print_model_prompt(idx, continue_prompt, tokenizer)
        else:
            n_toks = count_tokens(tokenizer, continue_prompt)
            kind = "re-run empty output" if item["retry_empty_output"] else "queued"
            print(
                f"[{idx}] {row.get('behavior_id', '')} {kind} ({n_toks} tokens)",
                flush=True,
            )
        pending.append({**item, "continue_prompt": continue_prompt})

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
            reasoning_text, content = reasoning_parser.parse_non_stream(stitched)
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
            upsert_jsonl_row(output_path, record)
            n_run += 1
            if not output:
                print(
                    f"[{idx}] {row.get('behavior_id', '')} still has empty output "
                    "(will retry on the next run)",
                    flush=True,
                )
            else:
                print(
                    f"[{idx}] {row.get('behavior_id', '')} hijack reasoning done",
                    flush=True,
                )
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} generations, skipped {n_skip} -> {output_path}")
    print(
        "Next:\n"
        "  python scripts/harmbench/contextual_without_context/qwen3-4b/generate_bypass_reasoning.py "
        f"--input {output_path}"
    )


if __name__ == "__main__":
    main()

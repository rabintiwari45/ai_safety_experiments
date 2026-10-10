"""
Run Qwen3 on HarmBench prompts with enable_thinking=False (no bypass prefix).

Reads the same bypass JSONL as other hijack scripts but only uses each row's
user prompt. The chat template adds an empty closed think block; generation
continues on the visible (non-reasoning) path.

Usage (from ai_safety_experiments/):
    python scripts/hijack_reasoninig/run_bypass_inference_output_prefix.py --print-prompt --limit 1
    python scripts/hijack_reasoninig/run_bypass_inference_output_prefix.py --dry-run
    python scripts/hijack_reasoninig/run_bypass_inference_output_prefix.py --limit 1
    python scripts/hijack_reasoninig/run_bypass_inference_output_prefix.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR / "misc"))
from common import configure_model_hub_env

configure_model_hub_env()

from transformers import AutoTokenizer

import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser

from common import (
    BASELINE_JSON,
    HIJACK_BYPASS_DEEPSEEKV3,
    HIJACK_PROMPT_ONLY_NO_THINK_INFER,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    count_tokens,
    load_infer_resume_state,
    load_records,
    upsert_jsonl_row,
)

DEFAULT_INPUT = HIJACK_BYPASS_DEEPSEEKV3
DEFAULT_OUTPUT = HIJACK_PROMPT_ONLY_NO_THINK_INFER
DEFAULT_BASELINE = BASELINE_JSON


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3 inference from user prompt only (enable_thinking=False, "
            "no bypass output prefix)"
        )
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
        help="Build prompts and skip GPU inference",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print exact llm.generate prompt (implies --dry-run)",
    )
    return parser.parse_args()


def build_prompt(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def log_banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def print_final_model_prompt(model_prompt: str, tokenizer) -> None:
    """Print the exact string passed to llm.generate (for debugging)."""
    n_toks = count_tokens(tokenizer, model_prompt)
    print(
        f"\n{'=' * 12} FINAL PROMPT SENT TO MODEL "
        f"(llm.generate, {n_toks} tokens) {'=' * 12}",
        flush=True,
    )
    print(
        model_prompt,
        end="" if model_prompt.endswith("\n") else "\n",
        flush=True,
    )
    print(f"{'=' * 12} END FINAL PROMPT {'=' * 12}", flush=True)


def main() -> None:
    args = parse_args()
    if args.print_prompt:
        args.dry_run = True
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

    print(f"HF_HOME={os.environ.get('HF_HOME', '')}", flush=True)
    print(
        "Mode: enable_thinking=False, user prompt only (no bypass prefix)",
        flush=True,
    )

    pending: list[dict] = []
    n_skip = 0
    for row in records:
        idx = row["index"]
        user_prompt = (row.get("prompt") or "").strip()
        original_reasoning = (row.get("original_reasoning") or "").strip()

        if idx in completed:
            print(f"[{idx}] skip: non-empty output already in {output_path}", flush=True)
            continue
        if idx in empty_output:
            print(
                f"[{idx}] re-run: row exists in output but output is empty",
                flush=True,
            )
        if not user_prompt:
            print(f"[{idx}] skip: empty prompt", flush=True)
            n_skip += 1
            continue

        model_prompt = build_prompt(tokenizer, user_prompt)
        print(
            f"\n[{idx}] {row.get('behavior_id', '')}  preparing model input",
            flush=True,
        )
        if args.dry_run or args.print_prompt:
            print_final_model_prompt(model_prompt, tokenizer)

        pending.append(
            {
                "row": row,
                "user_prompt": user_prompt,
                "original_reasoning": original_reasoning,
                "model_prompt": model_prompt,
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
        for item in pending:
            row = item["row"]
            user_prompt = item["user_prompt"]
            original_reasoning = item["original_reasoning"]
            model_prompt = item["model_prompt"]
            idx = row["index"]

            print(f"\n[{idx}] generating ...", flush=True)
            print_final_model_prompt(model_prompt, tokenizer)
            generated = llm.generate(
                prompt=model_prompt, sampling_params=sampling_params
            )
            generated_text = generated["text"]
            reasoning_text, content = reasoning_parser.parse_non_stream(
                generated_text
            )
            output = (content or generated_text).strip()

            record = {
                "index": idx,
                "behavior_id": row.get("behavior_id", ""),
                "category": row.get("category", ""),
                "prompt": user_prompt,
                "probe": row.get("probe", ""),
                "mode": "prompt_only_no_thinking",
                "enable_thinking": False,
                "original_reasoning": original_reasoning,
                "bypass_reasoning": row.get("bypass_reasoning", ""),
                "reasoning": (reasoning_text or "").strip(),
                "reasoning_tokens": count_tokens(
                    tokenizer, (reasoning_text or "")
                ),
                "output": output,
                "output_tokens": count_tokens(tokenizer, output),
                "k0_output": k0_by_index.get(idx, ""),
                "raw_text": generated_text,
                "generated_text": generated_text,
                "input_prompt": model_prompt,
                "skipped": False,
            }
            upsert_jsonl_row(output_path, record)
            n_run += 1

            log_banner("OUTPUT")
            print(output or "(empty)", flush=True)
            print(f"[{idx}] {row.get('behavior_id', '')} done", flush=True)
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} rows, skipped {n_skip} -> {output_path}")
    print(
        "Classify with:\n"
        f"  python scripts/misc/classify_harmbench.py --input {output_path}"
    )


if __name__ == "__main__":
    main()

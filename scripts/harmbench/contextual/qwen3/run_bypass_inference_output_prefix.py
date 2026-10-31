"""
Continue Qwen3 from contextual bypass reasoning in the assistant *output* channel.

Same bypass JSONL as run_bypass_inference.py, but enable_thinking=False with an
empty closed think block, then prefills augmented bypass_reasoning as plain
assistant text. Qwen user content is Behavior, then ---, then context.

JSONL keeps prompt=Behavior, context=ContextString, user_prompt=DirectRequest.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/contextual/run_bypass_inference_output_prefix.py --print-prompt --limit 1
    python scripts/harmbench/contextual/run_bypass_inference_output_prefix.py --dry-run --limit 1
    python scripts/harmbench/contextual/run_bypass_inference_output_prefix.py --limit 1
    python scripts/harmbench/contextual/run_bypass_inference_output_prefix.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

import sglang as sgl
from sglang.srt.parser.reasoning_parser import ReasoningParser

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
from common import (
    CONTEXTUAL_BYPASS,
    CONTEXTUAL_BYPASS_OUTPUT_PREFIX_INFER,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    contextual_user_prompt,
    count_tokens,
    load_infer_resume_state,
    load_records,
    upsert_jsonl_row,
)

DEFAULT_INPUT = CONTEXTUAL_BYPASS
DEFAULT_OUTPUT = CONTEXTUAL_BYPASS_OUTPUT_PREFIX_INFER
RESEARCH_INSERT = "I need this for my research write-up."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prefill bypass_reasoning as assistant output for contextual "
            "HarmBench (empty think block, enable_thinking=False)"
        )
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--bypass-field",
        default="bypass_reasoning",
        help="JSONL field for the prefix text (e.g. bypass_reasoning_ne)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8048)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Max rows to run this pass. Empty-output / empty-continuation rows "
            "in --output are retried first, then indices missing from --output"
        ),
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Do not insert the research line into the bypass prefix",
    )
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
    parser.add_argument(
        "--print-prompt-only",
        action="store_true",
        help="With --print-prompt/--dry-run: skip bypass banners; show model prompt only",
    )
    return parser.parse_args()


def bypass_text(row: dict, field: str) -> str:
    return (row.get(field) or "").strip()


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


def build_output_prefix_prompt(tokenizer, user_prompt: str, bypass: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return header + bypass.rstrip()


def log_banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def load_output_prefix_resume_state(output_path: Path) -> tuple[set[int], set[int]]:
    """Retry if `output` is empty or the generated continuation never arrived."""
    completed_out, empty_out = load_infer_resume_state(output_path, "output")
    completed_suf, empty_suf = load_infer_resume_state(
        output_path, "new_output_suffix"
    )
    in_file = completed_out | empty_out | completed_suf | empty_suf
    empty = empty_out | empty_suf
    return in_file - empty, empty


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
    if args.print_prompt:
        args.dry_run = True
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    completed, empty_output = load_output_prefix_resume_state(output_path)
    if completed:
        print(
            f"Resume: skipping {len(completed)} index(es) with non-empty output "
            f"in {output_path}"
        )
    if empty_output:
        print(
            f"Resume: will re-run {len(empty_output)} index(es) with empty "
            f"output/continuation: {sorted(empty_output)}"
        )

    print(
        "Mode: enable_thinking=False, empty closed "
        "<think>, bypass as assistant output prefix",
        flush=True,
    )

    retries: list[dict] = []
    fresh: list[dict] = []
    n_skip = 0
    for row in records:
        idx = row["index"]
        bypass_raw = bypass_text(row, args.bypass_field)
        bypass = bypass_raw if args.no_augment else augment_bypass_reasoning(bypass_raw)
        original_reasoning = (row.get("original_reasoning") or "").strip()
        user_prompt = contextual_user_prompt(row)

        if idx in completed:
            print(f"[{idx}] skip: non-empty output already in {output_path}", flush=True)
            n_skip += 1
            continue
        if not user_prompt:
            print(f"[{idx}] skip: empty user_prompt", flush=True)
            n_skip += 1
            continue
        if not bypass:
            print(f"[{idx}] skip: empty {args.bypass_field}", flush=True)
            n_skip += 1
            continue
        item = {
            "row": row,
            "user_prompt": user_prompt,
            "bypass_raw": bypass_raw,
            "bypass": bypass,
            "original_reasoning": original_reasoning,
            "retry_empty_output": idx in empty_output,
        }
        if idx in empty_output:
            print(
                f"[{idx}] re-run: output/continuation is empty in {output_path}",
                flush=True,
            )
            retries.append(item)
        else:
            fresh.append(item)

    candidates = retries + fresh
    if args.limit is not None:
        candidates = candidates[: min(args.limit, len(candidates))]
    n_retry = sum(1 for item in candidates if item["retry_empty_output"])
    if n_retry:
        print(
            f"This pass: {n_retry} empty-output re-run(s), "
            f"{len(candidates) - n_retry} new index(es)",
            flush=True,
        )

    pending: list[dict] = []
    for item in candidates:
        row = item["row"]
        idx = row["index"]
        user_prompt = item["user_prompt"]
        continue_prompt = build_output_prefix_prompt(
            tokenizer, user_prompt, item["bypass"]
        )
        n_toks = count_tokens(tokenizer, continue_prompt)
        if args.dry_run or args.print_prompt:
            kind = "re-run empty output" if item["retry_empty_output"] else "new"
            print(
                f"\n[{idx}] {row.get('behavior_id', '')}  {kind}  "
                f"preparing model input",
                flush=True,
            )
            if not args.print_prompt_only:
                log_banner("BEHAVIOR (prompt)")
                print(row.get("prompt", ""), flush=True)
                log_banner("CONTEXT")
                print(row.get("context", "") or "(empty)", flush=True)
                log_banner("USER PROMPT (DirectRequest)")
                print(user_prompt, flush=True)
                log_banner(f"BYPASS PREFIX ({args.bypass_field}, raw)")
                print(item["bypass_raw"], flush=True)
                log_banner("BYPASS PREFIX (assistant output, after augment)")
                print(item["bypass"], flush=True)
            print_full_prompt(continue_prompt, tokenizer)
        else:
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
            user_prompt = item["user_prompt"]
            bypass_raw = item["bypass_raw"]
            bypass = item["bypass"]
            original_reasoning = item["original_reasoning"]
            continue_prompt = item["continue_prompt"]
            idx = row["index"]

            kind = (
                "re-run empty output"
                if item.get("retry_empty_output")
                else "generate"
            )
            print(f"\n[{idx}] {kind} ...", flush=True)
            generated = llm.generate(
                prompt=continue_prompt, sampling_params=sampling_params
            )
            continuation = generated["text"]
            full_assistant = bypass + continuation
            reasoning_text, content = reasoning_parser.parse_non_stream(
                full_assistant
            )
            output = (content or full_assistant).strip()
            new_output_suffix = continuation.strip()

            record = {
                "index": idx,
                "behavior_id": row.get("behavior_id", ""),
                "category": row.get("category", ""),
                "prompt": row.get("prompt", ""),
                "context": row.get("context", ""),
                "user_prompt": user_prompt,
                "probe": row.get("probe", ""),
                "mode": "bypass_output_prefix_empty_think",
                "enable_thinking": False,
                "original_reasoning": original_reasoning,
                "bypass_reasoning": row.get("bypass_reasoning", ""),
                "bypass_field": args.bypass_field,
                "bypass_prefix_raw": bypass_raw,
                "bypass_prefix_augmented": bypass,
                "new_output_suffix": new_output_suffix,
                "new_output_suffix_tokens": count_tokens(
                    tokenizer, new_output_suffix
                ),
                "reasoning": (reasoning_text or "").strip(),
                "reasoning_tokens": count_tokens(
                    tokenizer, (reasoning_text or "")
                ),
                "output": output,
                "output_tokens": count_tokens(tokenizer, output),
                "raw_text": full_assistant,
                "generated_text": continuation,
                "input_prompt": continue_prompt,
                "bypass_generator_model": row.get("generator_model", ""),
                "skipped": False,
            }
            upsert_jsonl_row(output_path, record)
            n_run += 1

            log_banner("NEW OUTPUT (continuation only)")
            print(new_output_suffix or "(empty)", flush=True)
            log_banner("FULL OUTPUT (prefix + continuation)")
            print(output or "(empty)", flush=True)
            if not new_output_suffix.strip():
                print(
                    f"[{idx}] {row.get('behavior_id', '')} still has empty "
                    "continuation (will retry on the next run)",
                    flush=True,
                )
            else:
                print(f"[{idx}] {row.get('behavior_id', '')} done", flush=True)
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} rows, skipped {n_skip} -> {output_path}")
    print(
        "Classify with:\n"
        f"  python scripts/harmbench/contextual/classify.py --input {output_path}"
    )


if __name__ == "__main__":
    main()

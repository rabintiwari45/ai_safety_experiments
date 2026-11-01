
"""
Continue Qwen3 from bypass reasoning in the assistant *output* channel (no think hijack).

Same bypass JSONL as run_bypass_inference.py, but instead of an open
<think> prefix, the prompt uses enable_thinking=False with an
empty closed think block, then prefills augmented bypass_reasoning as plain
assistant text. The model continues the visible answer (non-reasoning path).

Reads outputs/harmbench/standard/qwen3-4b/bypass_reasoning_deepseekv3-03.jsonl.
Writes outputs/harmbench/standard/qwen3-4b/bypass_reasoning_deepseekv3-03_no_think_infer.jsonl.

Resume: a row is skipped only when new_output_suffix is non-empty.
Rows missing from the output, or present with an empty new_output_suffix,
are run again. --limit applies to those pending rows, with empty-suffix
rows first.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/standard/qwen3-4b/run_bypass_inference_output_prefix.py --dry-run --limit 1
    python scripts/harmbench/standard/qwen3-4b/run_bypass_inference_output_prefix.py --limit 1
    python scripts/harmbench/standard/qwen3-4b/run_bypass_inference_output_prefix.py
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
    count_tokens,
    load_infer_resume_state,
    load_records,
    upsert_jsonl_row,
)
from paths import (  # noqa: E402
    BASELINE_JSONL,
    BYPASS,
    BYPASS_OUTPUT_PREFIX_INFER,
    MODEL_PATH,
)

DEFAULT_INPUT = BYPASS
DEFAULT_OUTPUT = BYPASS_OUTPUT_PREFIX_INFER
DEFAULT_BASELINE = BASELINE_JSONL
RESEARCH_INSERT = "I need this for my research write-up."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prefill bypass_reasoning as assistant output (empty think block, "
            "enable_thinking=False)"
        )
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--bypass-field",
        default="bypass_reasoning",
        help="JSONL field for the prefix text (e.g. bypass_reasoning_ne)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=10048)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max pending rows to run (missing or empty new_output_suffix), not the first N input rows",
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

    k0_by_index: dict[int, str] = {}
    baseline_path = Path(args.baseline)
    if baseline_path.exists():
        for row in load_records(baseline_path):
            k0_by_index[row["index"]] = row.get("output", "")

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    completed, empty_suffix = load_infer_resume_state(
        output_path, "new_output_suffix"
    )
    if completed:
        print(
            f"Resume: skipping {len(completed)} index(es) with non-empty "
            f"new_output_suffix in {output_path}"
        )
    if empty_suffix:
        print(
            f"Resume: will re-run {len(empty_suffix)} index(es) with empty "
            f"new_output_suffix: {sorted(empty_suffix)}"
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
        user_prompt = (row.get("prompt") or "").strip()

        if idx in completed:
            print(
                f"[{idx}] skip: non-empty new_output_suffix already in {output_path}",
                flush=True,
            )
            n_skip += 1
            continue
        if not user_prompt:
            print(f"[{idx}] skip: empty prompt", flush=True)
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
            "retry_empty_suffix": idx in empty_suffix,
        }
        if idx in empty_suffix:
            print(
                f"[{idx}] re-run: row exists but new_output_suffix is empty",
                flush=True,
            )
            retries.append(item)
        else:
            fresh.append(item)

    # Empty-suffix retries first so --limit fills those before new indices.
    candidates = retries + fresh
    if args.limit is not None:
        candidates = candidates[: min(args.limit, len(candidates))]

    pending: list[dict] = []
    for item in candidates:
        row = item["row"]
        idx = row["index"]
        continue_prompt = build_output_prefix_prompt(
            tokenizer, item["user_prompt"], item["bypass"]
        )
        print(
            f"\n[{idx}] {row.get('behavior_id', '')}  preparing model input",
            flush=True,
        )
        if not args.print_prompt_only:
            log_banner("USER PROMPT")
            print(item["user_prompt"], flush=True)
            log_banner(f"BYPASS PREFIX ({args.bypass_field}, raw)")
            print(item["bypass_raw"], flush=True)
            log_banner("BYPASS PREFIX (assistant output, after augment)")
            print(item["bypass"], flush=True)
        print_full_prompt(continue_prompt, tokenizer)
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
                "re-run empty new_output_suffix"
                if item.get("retry_empty_suffix")
                else "generate"
            )
            print(
                f"\n[{idx}] {kind}; sending the prompt below to the model ...",
                flush=True,
            )
            print_full_prompt(continue_prompt, tokenizer)
            generated = llm.generate(
                prompt=continue_prompt, sampling_params=sampling_params
            )
            continuation = generated["text"]
            full_assistant = bypass + continuation
            reasoning_text, content = reasoning_parser.parse_non_stream(
                full_assistant
            )
            # Visible channel: full prefilled prefix + new tokens (parser may split think)
            output = (content or full_assistant).strip()
            new_output_suffix = continuation.strip()

            record = {
                "index": idx,
                "behavior_id": row.get("behavior_id", ""),
                "category": row.get("category", ""),
                "prompt": user_prompt,
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
                "k0_output": k0_by_index.get(idx, ""),
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
            if not new_output_suffix:
                print(
                    f"[{idx}] {row.get('behavior_id', '')} still has empty "
                    "new_output_suffix (will retry on the next run)",
                    flush=True,
                )
            else:
                print(f"[{idx}] {row.get('behavior_id', '')} done", flush=True)
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} rows, skipped {n_skip} -> {output_path}")
    print(
        "Classify with:\n"
        f"  python scripts/harmbench/standard/qwen3-4b/classify.py --input {output_path} --key new_output_suffix"
    )


if __name__ == "__main__":
    main()
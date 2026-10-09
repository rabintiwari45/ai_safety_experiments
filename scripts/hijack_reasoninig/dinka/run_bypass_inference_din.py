"""
Run Qwen3 thinking inference from Dinka hijack bypass traces.

Same as run_bypass_inference_ne.py but uses prompt_din and bypass_reasoning_din
from the translated bypass JSONL (default: outputs/hijack_reasoninig/dinka/
hijack_reasoninig_bypass_deepseekv3_din.jsonl).

Usage (from ai_safety_experiments/):
    python scripts/hijack_reasoninig/dinka/run_bypass_inference_din.py --print-prompt --print-prompt-only --limit 1
    python scripts/hijack_reasoninig/dinka/run_bypass_inference_din.py --dry-run
    python scripts/hijack_reasoninig/dinka/run_bypass_inference_din.py --limit 1
    python scripts/hijack_reasoninig/dinka/run_bypass_inference_din.py

Resume (automatic): re-runs any row with missing or empty "output" in --output;
skips rows that already have non-empty output. Re-run the same command to fill gaps:

    python scripts/hijack_reasoninig/dinka/run_bypass_inference_din.py \\
        --input outputs/hijack_reasoninig/dinka/hijack_reasoning_bypass_deepseekv3-03_probe_dinka.jsonl \\
        --output outputs/hijack_reasoninig/dinka/hijack_reasoning_bypass_deepseekv3-03_probe_dinka_infer.jsonl
"""

from __future__ import annotations

import argparse
import re
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
    HIJACK_DINKA_PROBE_DINKA,
    HIJACK_DINKA_PROBE_DINKA_INFER,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    count_tokens,
    load_infer_resume_state,
    load_records,
    upsert_jsonl_row,
)

DEFAULT_INPUT = HIJACK_DINKA_PROBE_DINKA
DEFAULT_OUTPUT = HIJACK_DINKA_PROBE_DINKA_INFER
DEFAULT_BASELINE = BASELINE_JSON
# Dinka research line (parallel to Nepali RESEARCH_INSERT in run_bypass_inference_ne.py)
RESEARCH_INSERT = "A cä mi cam abɛl ku thok cä rɛcdä"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen continue from Dinka bypass reasoning (open <think>)"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-tokens", type=int, default=9048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prompts and skip GPU inference",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print exact string passed to llm.generate (implies --dry-run)",
    )
    parser.add_argument(
        "--print-prompt-only",
        action="store_true",
        help="With --dry-run/--print-prompt: omit bypass banners; show model prompt only",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run every input row (ignore existing non-empty output in --output)",
    )
    return parser.parse_args()


def user_prompt_dinka(row: dict) -> str:
    return (row.get("prompt_dinka") or row.get("prompt_din") or "").strip()


def bypass_text_dinka(row: dict) -> str:
    return (
        row.get("bypass_reasoning_dinka") or row.get("bypass_reasoning_din") or ""
    ).strip()


def augment_bypass_reasoning(text: str) -> str:
    """Split on sentence boundaries; append Dinka research line to second-to-last."""
    raw = (text or "").strip()
    if not raw:
        return raw
    parts = [p.strip() for p in re.split(r"[.!?]+", raw) if p.strip()]
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
    if args.overwrite:
        completed = set()
        empty_output = set()
        print("Overwrite: re-running all input rows (ignoring existing output)", flush=True)
    elif completed:
        print(
            f"Resume: skipping {len(completed)} index(es) with non-empty output "
            f"in {output_path}",
            flush=True,
        )
    never_run: set[int] = set()
    pending: list[dict] = []
    n_skip = 0
    for row in records:
        idx = row["index"]
        prompt_din = user_prompt_dinka(row)
        bypass_raw = bypass_text_dinka(row)
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
        elif not args.overwrite:
            never_run.add(idx)
        if not prompt_din:
            print(f"[{idx}] skip: empty prompt_din", flush=True)
            n_skip += 1
            continue
        if not bypass:
            print(f"[{idx}] skip: empty bypass_reasoning_din", flush=True)
            n_skip += 1
            continue

        continue_prompt = build_continue_prompt(tokenizer, prompt_din, bypass)
        print(
            f"\n[{idx}] {row.get('behavior_id', '')}  preparing model input",
            flush=True,
        )
        if not args.print_prompt_only:
            log_banner("USER PROMPT (Dinka)")
            print(prompt_din, flush=True)
            log_banner("BYPASS_DINKA (raw from file)")
            print(bypass_raw, flush=True)
            log_banner(
                "BYPASS_DINKA IN <think> "
                "(augmented, </think> not closed)"
            )
            print(bypass, flush=True)
        print_full_prompt(continue_prompt, tokenizer)

        pending.append(
            {
                "row": row,
                "prompt_din": prompt_din,
                "bypass_raw": bypass_raw,
                "bypass": bypass,
                "original_reasoning": original_reasoning,
                "continue_prompt": continue_prompt,
            }
        )

    if empty_output and not args.overwrite:
        print(
            f"Resume: will re-run {len(empty_output)} index(es) with empty output: "
            f"{sorted(empty_output)}",
            flush=True,
        )
    if never_run and not args.overwrite:
        print(
            f"Resume: will run {len(never_run)} index(es) not yet in output: "
            f"{sorted(never_run)}",
            flush=True,
        )
    if pending:
        print(
            f"Total to generate: {len(pending)} row(s)",
            flush=True,
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
            prompt_din = item["prompt_din"]
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
                "prompt": row.get("prompt", ""),
                "prompt_din": prompt_din,
                "probe": row.get("probe", ""),
                "original_reasoning": original_reasoning,
                "bypass_reasoning": row.get("bypass_reasoning", ""),
                "bypass_reasoning_din": bypass_raw,
                "bypass_reasoning_din_augmented": bypass,
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
                "translator_model": row.get("translator_model", ""),
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
        f"  python scripts/misc/classify_harmbench.py --input {output_path}"
    )


if __name__ == "__main__":
    main()

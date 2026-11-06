"""
Run Gemma 4 E4B Instruct from StrongREJECT probes.

Same condition as the Qwen run: enable_thinking=True, then the probe is
prefilled inside an open thought channel (`<|channel>thought`) and
`<channel|>` is not closed. The model continues from there.

Reads prompt + probe from outputs/strongreject/gemma4/probes.jsonl.
Writes outputs/strongreject/gemma4/probes_reasoning.jsonl.

Resume: rows already in the output with a non-empty "output" are skipped.
Rows missing from the output, or present with an empty "output", are run
again. --limit applies to those pending rows, with empty-output rows first.

Usage (from ai_safety_experiments/):
    python scripts/strongreject/gemma4/run_inference.py --dry-run --limit 1
    python scripts/strongreject/gemma4/run_inference.py --limit 1
    python scripts/strongreject/gemma4/run_inference.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

import sglang as sgl

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    count_tokens,
    load_infer_resume_state,
    load_records,
    upsert_jsonl_row,
)
from paths import (  # noqa: E402
    BASELINE_JSONL,
    GEMMA_SAMPLING_PARAMS,
    MODEL_PATH,
    PROBE_INFER,
    PROBES,
    chat_header,
    join_open_thought,
    split_thought,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma 4 E4B continue from StrongREJECT probes (open thought channel)"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--probes", default=str(PROBES))
    parser.add_argument("--baseline", default=str(BASELINE_JSONL))
    parser.add_argument("--output", default=str(PROBE_INFER))
    parser.add_argument("--max-new-tokens", type=int, default=8048)
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
    return parser.parse_args()


def probe_text(row: dict) -> str:
    return (row.get("probe") or row.get("generator_raw") or "").strip()


def build_continue_prompt(tokenizer, user_prompt: str, probe: str) -> str:
    return chat_header(
        tokenizer, user_prompt, enable_thinking=True
    ) + join_open_thought(probe)


def log_banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def print_full_prompt(continue_prompt: str, tokenizer) -> None:
    n_toks = count_tokens(tokenizer, continue_prompt)
    print(
        f"\n{'=' * 12} FULL PROMPT SENT TO GEMMA "
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


def main() -> None:
    args = parse_args()
    probe_path = Path(args.probes)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(probe_path)

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

    retries: list[dict] = []
    fresh: list[dict] = []
    for row in records:
        idx = row["index"]
        probe = probe_text(row)
        if idx in completed:
            print(
                f"[{idx}] skip: non-empty output already in {output_path}",
                flush=True,
            )
            continue
        if row.get("skipped") or not probe:
            print(f"[{idx}] skip: empty probe", flush=True)
            continue
        item = {"row": row, "probe": probe, "retry_empty_output": idx in empty_output}
        if idx in empty_output:
            print(
                f"[{idx}] re-run: row exists in output but output is empty",
                flush=True,
            )
            retries.append(item)
        else:
            fresh.append(item)

    # Empty-output retries first so --limit fills those before new indices.
    candidates = retries + fresh
    if args.limit is not None:
        candidates = candidates[: min(args.limit, len(candidates))]

    pending: list[dict] = []
    for item in candidates:
        row = item["row"]
        probe = item["probe"]
        idx = row["index"]
        continue_prompt = build_continue_prompt(tokenizer, row["prompt"], probe)
        print(f"\n[{idx}] {row.get('behavior_id', '')}  preparing model input", flush=True)
        log_banner("USER PROMPT")
        print(row["prompt"], flush=True)
        log_banner("PROBE IN <|channel>thought (<channel|> not closed)")
        print(probe, flush=True)
        print_full_prompt(continue_prompt, tokenizer)
        pending.append({**item, "continue_prompt": continue_prompt})

    if args.dry_run:
        print(f"\nDry-run: {len(pending)} prompt(s) printed; not loading the GPU")
        return

    if not pending:
        print("Nothing to generate.")
        return

    sampling_params = dict(GEMMA_SAMPLING_PARAMS)
    sampling_params["max_new_tokens"] = args.max_new_tokens

    print(f"Loading engine from {args.model_path} ...")
    llm = sgl.Engine(
        model_path=args.model_path,
        mem_fraction_static=0.85,
    )

    n_run = 0
    try:
        for item in pending:
            row = item["row"]
            probe = item["probe"]
            continue_prompt = item["continue_prompt"]
            idx = row["index"]

            kind = "re-run empty output" if item.get("retry_empty_output") else "generate"
            print(
                f"\n[{idx}] {kind}; sending the prompt above to the model ...",
                flush=True,
            )
            print_full_prompt(continue_prompt, tokenizer)
            generated = llm.generate(
                prompt=continue_prompt, sampling_params=sampling_params
            )
            continuation = generated["text"]
            new_reasoning, output = split_thought(continuation)
            stitched = join_open_thought(probe) + continuation

            log_banner("RAW MODEL CONTINUATION")
            print(continuation if continuation else "(empty)", flush=True)
            log_banner("OUTPUT")
            print(output if output else "(empty)", flush=True)

            record = {
                "index": idx,
                "behavior_id": row.get("behavior_id", ""),
                "category": row.get("category", ""),
                "source": row.get("source", ""),
                "model": args.model_path,
                "prompt": row["prompt"],
                "probe": probe,
                "skipped": False,
                "new_reasoning": new_reasoning,
                "new_reasoning_tokens": count_tokens(tokenizer, new_reasoning),
                "reasoning": new_reasoning,
                "output": output,
                "k0_output": k0_by_index.get(idx, ""),
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
                    f"[{idx}] {row.get('behavior_id', '')} probe continuation done",
                    flush=True,
                )
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} generations -> {output_path}")
    print(
        "Classify with:\n"
        f"  python scripts/strongreject/gemma4/classify.py --input {output_path}"
    )


if __name__ == "__main__":
    main()

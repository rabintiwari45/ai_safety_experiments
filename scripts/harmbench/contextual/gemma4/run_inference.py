"""
Run Gemma 4 E4B Instruct from HarmBench contextual probes.

User content is the behavior, then ---, then the context passage.
enable_thinking=True, then the probe is prefilled inside an open thought
channel (`<|channel>thought`) and `<channel|>` is not closed.

Reads outputs/harmbench/contextual/gemma4/probes.jsonl.
Writes outputs/harmbench/contextual/gemma4/probes_reasoning.jsonl.
JSONL keeps prompt=Behavior and context=ContextString.

Usage (from ai_safety_experiments/):
    python scripts/harmbench/contextual/gemma4/run_inference.py --dry-run --limit 1
    python scripts/harmbench/contextual/gemma4/run_inference.py --limit 1
    python scripts/harmbench/contextual/gemma4/run_inference.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

import sglang as sgl

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    contextual_user_prompt,
    count_tokens,
    load_completed_ids,
    load_records,
)
from paths import (  # noqa: E402
    CONTEXTUAL_PROBES,
    GEMMA_SAMPLING_PARAMS,
    MODEL_PATH,
    PROBE_INFER,
    chat_header,
    join_open_thought,
    split_thought,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gemma 4 E4B continue from contextual probes (open thought channel)"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--probes", default=str(CONTEXTUAL_PROBES))
    parser.add_argument("--output", default=str(PROBE_INFER))
    parser.add_argument("--max-new-tokens", type=int, default=8048)
    parser.add_argument("--limit", type=int, default=None)
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
        probe = probe_text(row)
        user_prompt = contextual_user_prompt(row)
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
        if not user_prompt or not (row.get("context") or "").strip():
            print(f"[{idx}] skip: empty behavior or context", flush=True)
            continue
        continue_prompt = build_continue_prompt(tokenizer, user_prompt, probe)
        print(f"\n[{idx}] {row.get('behavior_id', '')}  preparing model input", flush=True)
        log_banner("BEHAVIOR (prompt)")
        print(row.get("prompt", ""), flush=True)
        log_banner("CONTEXT")
        print(row.get("context", ""), flush=True)
        log_banner("PROBE IN <|channel>thought (<channel|> not closed)")
        print(probe, flush=True)
        print_full_prompt(continue_prompt, tokenizer)
        pending.append(
            {
                "row": row,
                "probe": probe,
                "user_prompt": user_prompt,
                "continue_prompt": continue_prompt,
            }
        )

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
        with output_path.open("a", encoding="utf-8") as out_f:
            for item in pending:
                row = item["row"]
                probe = item["probe"]
                user_prompt = item["user_prompt"]
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
                    "model": args.model_path,
                    "prompt": row.get("prompt", ""),
                    "context": row.get("context", ""),
                    "user_prompt": user_prompt,
                    "probe": probe,
                    "skipped": False,
                    "new_reasoning": new_reasoning,
                    "new_reasoning_tokens": count_tokens(tokenizer, new_reasoning),
                    "reasoning": new_reasoning,
                    "output": output,
                    "raw_text": stitched,
                    "generated_text": continuation,
                    "input_prompt": continue_prompt,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                print(
                    f"[{idx}] {row.get('behavior_id', '')} probe continuation done",
                    flush=True,
                )
    finally:
        llm.shutdown()

    print(f"\nWrote {n_run} generations -> {output_path}")
    print(
        "Classify with:\n"
        f"  python scripts/harmbench/contextual/gemma4/classify.py --input {output_path}"
    )


if __name__ == "__main__":
    main()

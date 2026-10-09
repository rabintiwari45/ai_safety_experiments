"""
Run Qwen3 on Nepali HarmBench prompts, with or without the Nepali think probe.

Probe (default):
    {chat template}
    <think>
    {Nepali generator_raw, last words trimmed}

No-probe:
    {chat template, enable_thinking=True}

Reads outputs/translation/dynamic_nepali.jsonl.
Writes a new JSONL so earlier runs are not overwritten.

Usage (from harmbench/):
    python scripts/translation/run_nepali.py --dry-run
    python scripts/translation/run_nepali.py --limit 1
    python scripts/translation/run_nepali.py
    python scripts/translation/run_nepali.py --no-probe
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "misc"))
from common import (
    BASELINE_JSON,
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    TRANSLATION_JSONL,
    TRANSLATION_NOPROBE_JSONL,
    TRANSLATION_PROBE_JSONL,
    count_tokens,
    load_completed_ids,
    load_records,
)

DEFAULT_INPUT = TRANSLATION_JSONL
DEFAULT_BASELINE = BASELINE_JSON


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen Nepali prompts with optional Nepali think-opener probe"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument(
        "--output",
        default=None,
        help="JSONL path (default: *_probe.jsonl or *_noprobe.jsonl)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--trim-last-words",
        type=int,
        default=7,
        help="Drop this many words from the end of Nepali generator_raw",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Send only the Nepali user prompt (no think-opener)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full prompt for the first row; do not load the GPU",
    )
    return parser.parse_args()


def nepali_prompt(row: dict) -> str:
    return (row.get("prompt_ne") or row.get("prompt") or "").strip()


def opener_text(row: dict) -> str:
    return (row.get("generator_raw") or row.get("prefix") or "").strip()


def drop_last_words(text: str, n: int) -> str:
    words = text.split()
    if n <= 0 or not words:
        return text
    keep = max(1, len(words) - n)
    return " ".join(words[:keep])


def join_open_think(opener: str) -> str:
    return f"<think>\n{opener.rstrip()}"


def build_prompt(tokenizer, user_prompt: str, opener: str | None) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if opener:
        return header + join_open_think(opener)
    return header


def new_reasoning_from_continuation(continuation: str) -> str:
    close = continuation.find("</think>")
    if close == -1:
        return continuation
    return continuation[:close]


def log_banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def print_full_prompt(model_prompt: str, tokenizer) -> None:
    n_toks = count_tokens(tokenizer, model_prompt)
    print(
        f"\n{'=' * 12} FULL PROMPT SENT TO MODEL "
        f"({n_toks} tokens) {'=' * 12}",
        flush=True,
    )
    print(
        model_prompt,
        end="" if model_prompt.endswith("\n") else "\n",
        flush=True,
    )
    print(f"{'=' * 12} END FULL PROMPT {'=' * 12}", flush=True)
    print(f"prompt tail repr: {model_prompt[-80:]!r}", flush=True)


def debug_print_input(
    *,
    idx: int,
    behavior_id: str,
    user_prompt: str,
    opener_raw: str,
    opener: str,
    model_prompt: str,
    tokenizer,
    use_probe: bool,
    trim_last_words: int,
) -> None:
    mode = "Nepali prompt + Nepali probe" if use_probe else "Nepali prompt only"
    print(f"\n[{idx}] {behavior_id}  preparing {mode}", flush=True)
    log_banner("USER PROMPT (Nepali)")
    print(user_prompt, flush=True)
    if use_probe:
        log_banner(f"GENERATOR RAW (Nepali, before dropping last {trim_last_words} words)")
        print(opener_raw if opener_raw else "(empty)", flush=True)
        log_banner("THINKING PREFIX (trimmed, </think> not closed)")
        print(opener if opener else "(empty)", flush=True)
    else:
        log_banner("PROBE")
        print("(none)", flush=True)
    print_full_prompt(model_prompt, tokenizer)


def main() -> None:
    args = parse_args()
    use_probe = not args.no_probe
    input_path = Path(args.input)
    output_path = Path(
        args.output
        or (TRANSLATION_NOPROBE_JSONL if args.no_probe else TRANSLATION_PROBE_JSONL)
    )
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
    print(
        f"Mode: {'Nepali prompt + Nepali probe' if use_probe else 'Nepali prompt only'}",
        flush=True,
    )

    completed = load_completed_ids(output_path)
    if completed:
        print(f"Resuming: {len(completed)} rows already in {output_path}")

    pending: list[dict] = []
    for row in records:
        idx = row["index"]
        user_prompt = nepali_prompt(row)
        opener_raw = opener_text(row)
        opener = drop_last_words(opener_raw, args.trim_last_words) if use_probe else ""
        if idx in completed:
            print(
                f"[{idx}] already in {output_path}; skip generate. "
                "Delete that file or pass --output to rerun.",
                flush=True,
            )
            continue
        if not user_prompt:
            print(f"[{idx}] skip: empty Nepali prompt", flush=True)
            continue
        if use_probe and not opener_raw:
            print(f"[{idx}] skip: empty Nepali probe", flush=True)
            continue
        model_prompt = build_prompt(
            tokenizer, user_prompt, opener if use_probe else None
        )
        debug_print_input(
            idx=idx,
            behavior_id=row.get("behavior_id", ""),
            user_prompt=user_prompt,
            opener_raw=opener_raw,
            opener=opener,
            model_prompt=model_prompt,
            tokenizer=tokenizer,
            use_probe=use_probe,
            trim_last_words=args.trim_last_words,
        )
        pending.append(row)

    if args.dry_run:
        print(f"\nDry-run: {len(pending)} prompt(s) printed; not loading the GPU")
        print(f"Would write -> {output_path}")
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

                user_prompt = nepali_prompt(row)
                opener_raw = opener_text(row)
                opener = (
                    drop_last_words(opener_raw, args.trim_last_words)
                    if use_probe
                    else ""
                )
                if not user_prompt or (use_probe and not opener_raw):
                    skipped = {
                        "index": idx,
                        "behavior_id": row.get("behavior_id", ""),
                        "category": row.get("category", ""),
                        "prompt": user_prompt,
                        "prompt_en": (row.get("prompt_en") or "").strip(),
                        "prefix": opener,
                        "generator_raw": opener_raw,
                        "skipped": True,
                        "skip_reason": (
                            "empty_prompt" if not user_prompt else "empty_opener"
                        ),
                        "probe": use_probe,
                        "reasoning": "",
                        "output": "",
                        "k0_output": k0_by_index.get(idx, ""),
                        "raw_text": "",
                    }
                    out_f.write(json.dumps(skipped, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_skip += 1
                    continue

                model_prompt = build_prompt(
                    tokenizer, user_prompt, opener if use_probe else None
                )
                print(
                    f"\n[{idx}] sending the prompt above to the model ...",
                    flush=True,
                )
                print_full_prompt(model_prompt, tokenizer)
                generated = llm.generate(
                    prompt=model_prompt, sampling_params=sampling_params
                )
                continuation = generated["text"]
                stitched = (
                    join_open_think(opener) + continuation
                    if use_probe
                    else continuation
                )
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
                    "prompt": user_prompt,
                    "prompt_en": (row.get("prompt_en") or "").strip(),
                    "prefix": opener,
                    "generator_raw": opener_raw,
                    "skipped": False,
                    "probe": use_probe,
                    "new_reasoning": new_reasoning.strip(),
                    "new_reasoning_tokens": count_tokens(tokenizer, new_reasoning),
                    "reasoning": (reasoning_text or "").strip(),
                    "output": output,
                    "k0_output": k0_by_index.get(idx, ""),
                    "raw_text": stitched,
                    "generated_text": continuation,
                    "input_prompt": model_prompt,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                print(
                    f"[{idx}] {row.get('behavior_id', '')} "
                    f"nepali {'probe' if use_probe else 'no-probe'} done",
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

"""Shared probe-continuation loop for the Nepali HarmBench standard runs.

Each model script supplies the chat header, the open-think prefill, and how
to split the continuation. This module loads the Nepali prompt + probe,
skips finished rows, and writes one JSONL row per generation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer

import sglang as sgl

NEPALI_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = NEPALI_DIR.parent.parent
sys.path.insert(0, str(NEPALI_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))
from common import (  # noqa: E402
    count_tokens,
    load_infer_resume_state,
    load_records,
    upsert_jsonl_row,
)
from paths import NEPALI_PROBES  # noqa: E402


def join_open_think(probe: str) -> str:
    # No </think>. No trailing newline after a finished-looking sentence,
    # or Qwen often closes thinking immediately.
    return f"<think>\n{probe.rstrip()}"


def new_reasoning_from_continuation(continuation: str) -> str:
    close = continuation.find("</think>")
    if close == -1:
        return continuation
    return continuation[:close]


def repeated_region(text: str) -> tuple[int, int] | None:
    """Return (start, unit_length) when the tail is the same span copied many times.

    A hit means the model is stuck repeating one phrase. The caller keeps one
    copy and stops that generation.
    """
    min_unit = 8
    max_unit = 240
    min_repeats = 4
    min_covered = 160
    n = len(text)
    if n < min_covered:
        return None
    upper = min(max_unit, n // min_repeats)
    for length in range(min_unit, upper + 1):
        for slack in range(length):
            end = n - slack
            if end < length * min_repeats:
                break
            unit = text[end - length : end]
            if len(set(unit)) < 3:
                continue
            reps = 1
            pos = end - length
            while pos >= length and text[pos - length : pos] == unit:
                reps += 1
                pos -= length
            if reps >= min_repeats and reps * length >= min_covered:
                return pos, length
    return None


def qwen_parse(continuation: str, probe: str, reasoning_parser) -> dict:
    stitched = join_open_think(probe) + continuation
    reasoning_text, content = reasoning_parser.parse_non_stream(stitched)
    new_reasoning = new_reasoning_from_continuation(continuation)
    return {
        "new_reasoning": new_reasoning.strip(),
        "reasoning": (reasoning_text or "").strip(),
        "output": (content or "").strip(),
        "raw_text": stitched,
        "generated_text": continuation,
    }


def probe_text(row: dict) -> str:
    return (row.get("probe") or row.get("generator_raw") or "").strip()


def log_banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def print_full_prompt(continue_prompt: str, tokenizer, title: str) -> None:
    n_toks = count_tokens(tokenizer, continue_prompt)
    print(
        f"\n{'=' * 12} FULL PROMPT SENT TO {title} "
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


def generate_continuation(
    llm,
    prompt: str,
    sampling_params: dict,
    *,
    stop_on_repetition: bool,
) -> tuple[str, bool]:
    """Generate one continuation. Abort once the tail is the same span repeated."""
    if not stop_on_repetition:
        generated = llm.generate(prompt=prompt, sampling_params=sampling_params)
        return generated["text"], False
    return _stream_until_repeat(llm, prompt, sampling_params)


def _stream_until_repeat(
    llm, prompt: str, sampling_params: dict
) -> tuple[str, bool]:
    rid = f"nepali-repeat-{time.time_ns()}"
    stream = llm.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        stream=True,
        rid=rid,
    )
    text = ""
    stopped = False
    try:
        for chunk in stream:
            chunk_text = chunk.get("text")
            if chunk_text:
                # Non-incremental streaming sends the full text so far.
                # A shorter chunk that does not extend it is a delta.
                if not text or chunk_text.startswith(text):
                    text = chunk_text
                else:
                    text += chunk_text
            if stopped or repeated_region(text) is None:
                continue
            stopped = True
            print(
                "Repetition detected; cutting off this row and keeping the text.",
                flush=True,
            )
            llm.tokenizer_manager.abort_request(rid)
    finally:
        stream.close()

    # Keep the repeated span that was already produced. Further tokens are
    # not generated, and the caller does not run the row again.
    return text, stopped


def build_parser(
    *,
    description: str,
    model_path: str,
    default_output: Path,
    default_max_new_tokens: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model-path", default=model_path)
    parser.add_argument("--probes", default=str(NEPALI_PROBES))
    parser.add_argument("--output", default=str(default_output))
    parser.add_argument("--baseline", default="")
    parser.add_argument("--max-new-tokens", type=int, default=default_max_new_tokens)
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
    return parser


def run_probe_inference(
    *,
    description: str,
    model_path: str,
    default_output: Path,
    sampling_base: dict,
    default_max_new_tokens: int,
    build_continue_prompt,
    parse_generation,
    engine_kwargs: dict,
    parser_name: str | None,
    full_prompt_title: str,
    think_banner: str,
    stop_on_repetition: bool = False,
) -> None:
    args = build_parser(
        description=description,
        model_path=model_path,
        default_output=default_output,
        default_max_new_tokens=default_max_new_tokens,
    ).parse_args()

    probe_path = Path(args.probes)
    output_path = Path(args.output)
    if not probe_path.exists():
        raise FileNotFoundError(probe_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(probe_path)

    k0_by_index: dict[int, str] = {}
    if args.baseline:
        baseline_path = Path(args.baseline)
        if baseline_path.exists():
            for row in load_records(baseline_path):
                k0_by_index[row["index"]] = row.get("output", "")

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    completed, empty_output = load_infer_resume_state(output_path)
    if stop_on_repetition and output_path.exists():
        # A cut-off repeat is finished. Do not generate that row again, even
        # when it has no answer.
        for saved in load_records(output_path):
            if not saved.get("repetition_stopped"):
                continue
            idx = saved.get("index")
            if idx is None:
                continue
            completed.add(int(idx))
            empty_output.discard(int(idx))
    if completed:
        print(
            f"Resume: skipping {len(completed)} finished index(es) in {output_path}"
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
                f"[{idx}] skip: already saved in {output_path}",
                flush=True,
            )
            continue
        if row.get("skipped") or not probe or not (row.get("prompt") or "").strip():
            print(f"[{idx}] skip: empty Nepali prompt or probe", flush=True)
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
        log_banner("USER PROMPT (Nepali)")
        print(row["prompt"], flush=True)
        log_banner(think_banner)
        print(probe, flush=True)
        print_full_prompt(continue_prompt, tokenizer, full_prompt_title)
        pending.append({**item, "continue_prompt": continue_prompt})

    if args.dry_run:
        print(f"\nDry-run: {len(pending)} prompt(s) printed; not loading the GPU")
        return

    if not pending:
        print("Nothing to generate.")
        return

    sampling_params = dict(sampling_base)
    sampling_params["max_new_tokens"] = args.max_new_tokens

    reasoning_parser = None
    if parser_name:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        reasoning_parser = ReasoningParser(parser_name)

    print(f"Loading engine from {args.model_path} ...")
    llm = sgl.Engine(model_path=args.model_path, **engine_kwargs)

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
            print_full_prompt(continue_prompt, tokenizer, full_prompt_title)
            continuation, repetition_stopped = generate_continuation(
                llm,
                continue_prompt,
                sampling_params,
                stop_on_repetition=stop_on_repetition,
            )
            parsed = parse_generation(continuation, probe, reasoning_parser)
            output = (parsed.get("output") or "").strip()

            log_banner("RAW MODEL CONTINUATION")
            print(continuation if continuation else "(empty)", flush=True)
            log_banner("THINKING")
            print((parsed.get("reasoning") or "").strip() or "(empty)", flush=True)
            log_banner("OUTPUT")
            print(output if output else "(empty)", flush=True)

            record = {
                "index": idx,
                "behavior_id": row.get("behavior_id", ""),
                "category": row.get("category", ""),
                "model": args.model_path,
                "language": "ne",
                "prompt": row["prompt"],
                "prompt_en": row.get("prompt_en", ""),
                "probe": probe,
                "probe_en": row.get("probe_en", ""),
                "skipped": False,
                "new_reasoning": parsed.get("new_reasoning") or "",
                "new_reasoning_tokens": count_tokens(
                    tokenizer, parsed.get("new_reasoning") or ""
                ),
                "reasoning": (parsed.get("reasoning") or "").strip(),
                "output": output,
                "k0_output": k0_by_index.get(idx, ""),
                "raw_text": parsed.get("raw_text") or "",
                "generated_text": parsed.get("generated_text") or continuation,
                "input_prompt": continue_prompt,
                "repetition_stopped": repetition_stopped,
            }
            upsert_jsonl_row(output_path, record)
            n_run += 1
            if repetition_stopped:
                print(
                    f"[{idx}] {row.get('behavior_id', '')} cut off and kept "
                    "with the repeated text; it will not be generated again",
                    flush=True,
                )
            elif not output:
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

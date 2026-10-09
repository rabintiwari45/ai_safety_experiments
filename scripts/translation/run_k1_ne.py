"""
Nepali k=1: continue Qwen thinking after prefix + 128 new_reasoning tokens
plus the GPT Nepali probe (generator_k1_raw).

User message = prompt_ne.
Think (</think> not closed):

    {chat template}
    <think>
    {original Nepali prefix}
    {first 128 tokens of old new_reasoning}
    {generator_k1_raw}

Reads:
  - classified traces (prefix + new_reasoning)
  - dynamic_nepali_k1_probes_ne.jsonl (generator_k1_raw)

Writes a new JSONL. Does not overwrite earlier Nepali runs.

Usage (from harmbench/):
    python scripts/translation/run_k1_ne.py --dry-run
    python scripts/translation/run_k1_ne.py --limit 1
    python scripts/translation/run_k1_ne.py
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
    MODEL_PATH,
    THINKING_SAMPLING_PARAMS,
    TRANSLATION_K1_NE_JSONL,
    TRANSLATION_K1_PROBES_NE,
    TRANSLATION_OUTPUT_EN_CLASSIFIED_EN,
    count_tokens,
    load_completed_ids,
    load_records,
)

DEFAULT_TRACES = TRANSLATION_OUTPUT_EN_CLASSIFIED_EN
DEFAULT_PROBES = TRANSLATION_K1_PROBES_NE
DEFAULT_OUTPUT = TRANSLATION_K1_NE_JSONL
DEFAULT_STEP = 128
DEVANAGARI = ("\u0900", "\u097f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qwen Nepali k=1: prefix + 128 new_reasoning + generator_k1_raw"
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument(
        "--traces",
        default=str(DEFAULT_TRACES),
        help="Classified JSONL with prefix and new_reasoning",
    )
    parser.add_argument(
        "--probes",
        default=str(DEFAULT_PROBES),
        help="Nepali k=1 probe JSONL with generator_k1_raw",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--max-new-tokens", type=int, default=4048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--strict-step",
        action="store_true",
        help="Skip rows whose new_reasoning is shorter than --step tokens",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full prompt for the first eligible row; do not load the GPU",
    )
    return parser.parse_args()


def has_devanagari(text: str) -> bool:
    return any(DEVANAGARI[0] <= c <= DEVANAGARI[1] for c in text)


def nepali_prompt(trace: dict, probe: dict) -> str:
    for row in (trace, probe):
        for key in ("prompt_ne", "prompt"):
            text = (row.get(key) or "").strip()
            if text and has_devanagari(text):
                return text
    return ""


def prefix_text(trace: dict, probe: dict) -> str:
    for row in (trace, probe):
        text = (row.get("prefix") or "").strip()
        if text:
            return text
    for row in (trace, probe):
        text = (row.get("generator_raw") or "").strip()
        if text:
            return text
    return ""


def new_reasoning_text(trace: dict, probe: dict) -> str:
    text = (trace.get("new_reasoning") or "").strip()
    if text:
        return text
    return (probe.get("new_reasoning_128") or probe.get("new_reasoning") or "").strip()


def k1_raw_text(probe: dict) -> str:
    return (probe.get("generator_k1_raw") or probe.get("k1_probe") or "").strip()


def records_by_index(path: Path) -> dict[int, dict]:
    out = {}
    for row in load_records(path):
        idx = row.get("index")
        if idx is not None:
            out[idx] = row
    return out


def cut_reasoning(
    tokenizer, reasoning: str, step: int, *, allow_short: bool
) -> tuple[str | None, int]:
    ids = tokenizer.encode(reasoning or "", add_special_tokens=False)
    n = len(ids)
    if n == 0:
        return None, 0
    if n < step and not allow_short:
        return None, n
    take = min(step, n)
    cut = tokenizer.decode(ids[:take], skip_special_tokens=False)
    return cut, n


def join_think(prefix: str, new_128: str) -> str:
    if not prefix:
        return new_128
    if not new_128:
        return prefix
    if prefix.endswith((" ", "\n")) or new_128.startswith((" ", "\n")):
        return prefix + new_128
    return f"{prefix} {new_128}"


def ensure_nl(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def join_k1_think(prefix: str, new_128: str, k1_raw: str) -> str:
    body = join_think(prefix, new_128)
    return f"<think>\n{ensure_nl(body)}{k1_raw.rstrip()}"


def build_continue_prompt(
    tokenizer,
    user_prompt: str,
    prefix: str,
    new_128: str,
    k1_raw: str,
) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    header = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return header + join_k1_think(prefix, new_128, k1_raw)


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
        f"\n{'=' * 12} FULL PROMPT SENT TO QWEN3-8B "
        f"({n_toks} tokens) {'=' * 12}",
        flush=True,
    )
    print(
        continue_prompt,
        end="" if continue_prompt.endswith("\n") else "\n",
        flush=True,
    )
    print(f"{'=' * 12} END FULL PROMPT {'=' * 12}", flush=True)
    print(f"prompt tail repr: {continue_prompt[-120:]!r}", flush=True)


def debug_print_input(
    *,
    idx: int,
    behavior_id: str,
    user_prompt: str,
    prefix: str,
    new_128: str,
    k1_raw: str,
    continue_prompt: str,
    tokenizer,
    n_tokens: int,
    step: int,
) -> None:
    print(f"\n[{idx}] {behavior_id}  preparing Nepali k=1 model input", flush=True)
    log_banner("USER PROMPT (prompt_ne)")
    print(user_prompt, flush=True)
    log_banner("ORIGINAL PREFIX")
    print(prefix if prefix else "(empty)", flush=True)
    log_banner(f"OLD NEW_REASONING (first {step} tokens, source has {n_tokens})")
    print(new_128 if new_128 else "(empty)", flush=True)
    log_banner("K1 PROBE (generator_k1_raw)")
    print(k1_raw if k1_raw else "(empty)", flush=True)
    print_full_prompt(continue_prompt, tokenizer)


def skip_reason_for(
    *,
    user_prompt: str,
    prefix: str,
    new_128: str | None,
    k1_raw: str,
    probe: dict | None,
    n_tokens: int,
    step: int,
) -> str:
    if probe is None:
        return "missing_k1_probe_row"
    if probe.get("skipped") and not k1_raw:
        return probe.get("skip_reason") or "empty_k1_probe"
    if not user_prompt:
        return "missing_nepali_prompt"
    if not prefix:
        return "empty_prefix"
    if new_128 is None:
        return f"reasoning_shorter_than_step:{n_tokens}<{step}"
    if not k1_raw:
        return probe.get("skip_reason") or "empty_k1_probe"
    return ""


def main() -> None:
    args = parse_args()
    traces_path = Path(args.traces)
    probes_path = Path(args.probes)
    output_path = Path(args.output)
    if not traces_path.exists():
        raise FileNotFoundError(traces_path)
    if not probes_path.exists():
        raise FileNotFoundError(probes_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    traces = records_by_index(traces_path)
    probes = records_by_index(probes_path)
    indices = sorted(set(traces) | set(probes))
    if args.limit is not None:
        indices = indices[: args.limit]

    print(f"Loading tokenizer from {args.model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    allow_short = not args.strict_step

    completed = load_completed_ids(output_path)
    if completed:
        print(f"Resuming: {len(completed)} rows already in {output_path}")

    pending: list[int] = []
    prepared: dict[int, dict] = {}
    for idx in indices:
        if idx in completed:
            print(
                f"[{idx}] already in {output_path}; skip generate. "
                "Delete that file or pass --output to rerun.",
                flush=True,
            )
            continue
        trace = traces.get(idx, {})
        probe = probes.get(idx)
        user_prompt = nepali_prompt(trace, probe or {})
        prefix = prefix_text(trace, probe or {})
        new_r = new_reasoning_text(trace, probe or {})
        new_128, n_tokens = cut_reasoning(
            tokenizer, new_r, args.step, allow_short=allow_short
        )
        k1_raw = k1_raw_text(probe or {})
        reason = skip_reason_for(
            user_prompt=user_prompt,
            prefix=prefix,
            new_128=new_128,
            k1_raw=k1_raw,
            probe=probe,
            n_tokens=n_tokens,
            step=args.step,
        )
        prepared[idx] = {
            "trace": trace,
            "probe": probe or {},
            "user_prompt": user_prompt,
            "prefix": prefix,
            "new_128": new_128 or "",
            "n_tokens": n_tokens,
            "k1_raw": k1_raw,
            "skip_reason": reason,
        }
        if reason:
            print(f"[{idx}] skip: {reason}", flush=True)
            continue
        continue_prompt = build_continue_prompt(
            tokenizer, user_prompt, prefix, new_128, k1_raw
        )
        if args.dry_run:
            debug_print_input(
                idx=idx,
                behavior_id=(probe or trace).get("behavior_id", ""),
                user_prompt=user_prompt,
                prefix=prefix,
                new_128=new_128,
                k1_raw=k1_raw,
                continue_prompt=continue_prompt,
                tokenizer=tokenizer,
                n_tokens=n_tokens,
                step=args.step,
            )
        pending.append(idx)

    if args.dry_run:
        print(f"\nDry-run: {len(pending)} prompt(s) printed; not loading the GPU")
        print(f"Would write -> {output_path}")
        return

    to_write = [idx for idx in indices if idx not in completed and idx in prepared]
    if not to_write:
        print("Nothing to generate.")
        return

    sampling_params = dict(THINKING_SAMPLING_PARAMS)
    sampling_params["max_new_tokens"] = args.max_new_tokens

    llm = None
    reasoning_parser = None
    if pending:
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
            for idx in indices:
                if idx in completed:
                    continue
                item = prepared.get(idx)
                if item is None:
                    continue
                trace = item["trace"]
                probe = item["probe"]
                user_prompt = item["user_prompt"]
                prefix = item["prefix"]
                new_128 = item["new_128"]
                k1_raw = item["k1_raw"]
                n_tokens = item["n_tokens"]
                reason = item["skip_reason"]
                meta = probe or trace

                if reason:
                    skipped = {
                        "index": idx,
                        "behavior_id": meta.get("behavior_id", ""),
                        "category": meta.get("category", ""),
                        "prompt": user_prompt,
                        "prompt_en": (trace.get("prompt_en") or probe.get("prompt_en") or ""),
                        "prompt_ne": user_prompt,
                        "k": 1,
                        "step": args.step,
                        "skipped": True,
                        "skip_reason": reason,
                        "prefix": prefix,
                        "new_reasoning_128": new_128,
                        "new_reasoning_tokens_src": n_tokens,
                        "generator_k1_raw": k1_raw,
                        "k1_probe": k1_raw,
                        "reasoning": "",
                        "output": "",
                        "dynamic_output": (
                            trace.get("output") or probe.get("dynamic_output") or ""
                        ),
                        "output_en": trace.get("output_en") or probe.get("output_en") or "",
                        "k0_output": trace.get("k0_output") or probe.get("k0_output") or "",
                        "raw_text": "",
                    }
                    out_f.write(json.dumps(skipped, ensure_ascii=False) + "\n")
                    out_f.flush()
                    n_skip += 1
                    continue

                continue_prompt = build_continue_prompt(
                    tokenizer, user_prompt, prefix, new_128, k1_raw
                )
                print(
                    f"\n[{idx}] sending this full prompt to Qwen3-8B ...",
                    flush=True,
                )
                debug_print_input(
                    idx=idx,
                    behavior_id=meta.get("behavior_id", ""),
                    user_prompt=user_prompt,
                    prefix=prefix,
                    new_128=new_128,
                    k1_raw=k1_raw,
                    continue_prompt=continue_prompt,
                    tokenizer=tokenizer,
                    n_tokens=n_tokens,
                    step=args.step,
                )
                generated = llm.generate(
                    prompt=continue_prompt, sampling_params=sampling_params
                )
                continuation = generated["text"]
                stitched = join_k1_think(prefix, new_128, k1_raw) + continuation
                reasoning_text, content = reasoning_parser.parse_non_stream(stitched)
                output = (content or "").strip()
                new_reasoning = new_reasoning_from_continuation(continuation)

                log_banner("RAW MODEL CONTINUATION")
                print(continuation if continuation else "(empty)", flush=True)
                log_banner("OUTPUT")
                print(output if output else "(empty)", flush=True)

                record = {
                    "index": idx,
                    "behavior_id": meta.get("behavior_id", ""),
                    "category": meta.get("category", ""),
                    "prompt": user_prompt,
                    "prompt_en": (trace.get("prompt_en") or probe.get("prompt_en") or ""),
                    "prompt_ne": user_prompt,
                    "k": 1,
                    "step": args.step,
                    "skipped": False,
                    "prefix": prefix,
                    "new_reasoning_128": new_128,
                    "new_reasoning_tokens_src": n_tokens,
                    "generator_k1_raw": k1_raw,
                    "k1_probe": k1_raw,
                    "new_reasoning": new_reasoning.strip(),
                    "new_reasoning_tokens": count_tokens(tokenizer, new_reasoning),
                    "reasoning": (reasoning_text or "").strip(),
                    "output": output,
                    "dynamic_output": (
                        trace.get("output") or probe.get("dynamic_output") or ""
                    ),
                    "output_en": trace.get("output_en") or probe.get("output_en") or "",
                    "k0_output": trace.get("k0_output") or probe.get("k0_output") or "",
                    "raw_text": stitched,
                    "generated_text": continuation,
                    "input_prompt": continue_prompt,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_run += 1
                print(
                    f"[{idx}] {meta.get('behavior_id', '')} Nepali k=1 done",
                    flush=True,
                )
    finally:
        if llm is not None:
            llm.shutdown()

    print(f"\nWrote {n_run} generations, skipped {n_skip} -> {output_path}")
    print(
        "Translate output to English then classify_en, with explicit --input/--output."
    )


if __name__ == "__main__":
    main()

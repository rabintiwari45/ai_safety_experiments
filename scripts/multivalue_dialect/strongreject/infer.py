"""Resume loop for StrongREJECT dialect-probe inference.

Each model script builds its own chat prompt and engine, then calls
run_dialect. This module only handles the StrongREJECT dialect rows:

    prompt_african_american_vernacular / probe_african_american_vernacular
    prompt_colloquial_singapore        / probe_colloquial_singapore

The original English behavior stays in "prompt". The text sent to the model
is "user_prompt".
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    count_tokens,
    load_infer_resume_state,
    load_records,
    upsert_jsonl_row,
)

DIALECTS = {
    "african_american_vernacular": "probes_dialects_reasoning_african_american.jsonl",
    "colloquial_singapore": "probes_dialects_reasoning_singapore.jsonl",
}


def parse_args(
    *,
    description: str,
    model_path: str,
    probes: Path,
    baseline: Path,
    output_dir: Path,
    max_new_tokens: int,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dialect",
        required=True,
        choices=sorted(DIALECTS),
        help="Which transformed prompt and probe fields to send",
    )
    parser.add_argument("--model-path", default=model_path)
    parser.add_argument("--probes", default=str(probes))
    parser.add_argument("--baseline", default=str(baseline))
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL. Default: probes_dialects_reasoning_<dialect>.jsonl",
    )
    parser.add_argument("--max-new-tokens", type=int, default=max_new_tokens)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max pending rows (empty-output rows first), not the first N input rows",
    )
    parser.add_argument(
        "--empty-retries",
        type=int,
        default=2,
        help="Extra generations in this run when the parsed output is empty",
    )
    parser.add_argument(
        "--empty-only",
        action="store_true",
        help="Only re-run rows whose saved output is empty",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full prompt for pending rows; do not load the GPU",
    )
    args = parser.parse_args()
    if not args.output:
        args.output = str(output_dir / DIALECTS[args.dialect])
    return args


def field_text(row: dict, field: str) -> str:
    value = row.get(field)
    if isinstance(value, str):
        return value.strip()
    return ""


def log_banner(title: str) -> None:
    print(f"\n{'=' * 12} {title} {'=' * 12}", flush=True)


def print_full_prompt(continue_prompt: str, tokenizer, model_label: str) -> None:
    n_toks = count_tokens(tokenizer, continue_prompt)
    print(
        f"\n{'=' * 12} FULL PROMPT SENT TO {model_label} "
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


def collect_pending(
    *,
    records: list[dict],
    dialect: str,
    prompt_field: str,
    probe_field: str,
    output_path: Path,
    tokenizer,
    limit: int | None,
    empty_only: bool,
    build_continue_prompt,
    model_label: str,
    probe_channel: str,
) -> tuple[list[dict], int]:
    completed, empty_output = load_infer_resume_state(output_path)
    if completed:
        print(
            f"[{dialect}] resume: skipping {len(completed)} index(es) "
            f"with non-empty output in {output_path}",
            flush=True,
        )
    if empty_output:
        print(
            f"[{dialect}] resume: will re-run {len(empty_output)} index(es) "
            f"with empty output: {sorted(empty_output)}",
            flush=True,
        )

    retries: list[dict] = []
    fresh: list[dict] = []
    n_skip = 0
    for row in records:
        idx = row["index"]
        user_prompt = field_text(row, prompt_field)
        if not user_prompt:
            user_prompt = field_text(row, "prompt")
            if user_prompt:
                print(
                    f"[{dialect} {idx}] empty {prompt_field}; using English prompt",
                    flush=True,
                )
        probe = field_text(row, probe_field)

        if idx in completed:
            print(
                f"[{dialect} {idx}] skip: non-empty output already in {output_path}",
                flush=True,
            )
            n_skip += 1
            continue
        if empty_only and idx not in empty_output:
            print(
                f"[{dialect} {idx}] skip: --empty-only and output is not empty",
                flush=True,
            )
            n_skip += 1
            continue
        if row.get("skipped") or not probe or not user_prompt:
            print(
                f"[{dialect} {idx}] skip: empty {prompt_field} or {probe_field}",
                flush=True,
            )
            n_skip += 1
            continue

        item = {
            "row": row,
            "user_prompt": user_prompt,
            "probe": probe,
            "retry_empty_output": idx in empty_output,
        }
        if idx in empty_output:
            print(
                f"[{dialect} {idx}] re-run: row exists in output but output is empty",
                flush=True,
            )
            retries.append(item)
        else:
            fresh.append(item)

    candidates = retries + fresh
    if limit is not None:
        candidates = candidates[: min(limit, len(candidates))]

    pending: list[dict] = []
    for item in candidates:
        row = item["row"]
        idx = row["index"]
        continue_prompt = build_continue_prompt(
            tokenizer, item["user_prompt"], item["probe"]
        )
        print(
            f"\n[{dialect} {idx}] {row.get('behavior_id', '')}  preparing model input",
            flush=True,
        )
        log_banner(f"USER PROMPT ({prompt_field})")
        print(item["user_prompt"], flush=True)
        log_banner(f"PROBE IN {probe_channel} ({probe_field}, channel not closed)")
        print(item["probe"], flush=True)
        print_full_prompt(continue_prompt, tokenizer, model_label)
        pending.append({**item, "continue_prompt": continue_prompt})
    return pending, n_skip


def generate_pending(
    *,
    llm,
    parser,
    tokenizer,
    sampling_params,
    pending: list[dict],
    dialect: str,
    prompt_field: str,
    probe_field: str,
    output_path: Path,
    k0_by_index: dict[int, str],
    empty_retries: int,
    model_path: str,
    model_label: str,
    parse_generation,
) -> int:
    n_run = 0
    max_attempts = 1 + max(0, empty_retries)
    for item in pending:
        row = item["row"]
        user_prompt = item["user_prompt"]
        probe = item["probe"]
        continue_prompt = item["continue_prompt"]
        idx = row["index"]
        kind = "re-run empty output" if item["retry_empty_output"] else "generate"
        print(
            f"\n[{dialect} {idx}] {kind}: sending the prompt below to the model ...",
            flush=True,
        )
        print_full_prompt(continue_prompt, tokenizer, model_label)

        output = ""
        continuation = ""
        stitched = ""
        reasoning_text = ""
        new_reasoning = ""
        attempts_used = 0
        for attempt in range(1, max_attempts + 1):
            attempts_used = attempt
            if attempt > 1:
                print(
                    f"[{dialect} {idx}] empty output; running inference again "
                    f"({attempt - 1}/{empty_retries}) ...",
                    flush=True,
                )
            generated = llm.generate(
                prompt=continue_prompt, sampling_params=sampling_params
            )
            continuation = generated["text"]
            new_reasoning, reasoning_text, output, stitched = parse_generation(
                continuation, probe, parser
            )
            if output:
                break

        record = {
            "index": idx,
            "behavior_id": row.get("behavior_id", ""),
            "category": row.get("category", ""),
            "source": row.get("source", ""),
            "model": model_path,
            "dialect": dialect,
            "prompt": row.get("prompt", ""),
            "user_prompt": user_prompt,
            "prompt_field": prompt_field,
            "probe_field": probe_field,
            "probe": probe,
            "probe_english": field_text(row, "probe"),
            "skipped": False,
            "new_reasoning": new_reasoning,
            "new_reasoning_tokens": count_tokens(tokenizer, new_reasoning),
            "reasoning": reasoning_text,
            "output": output,
            "k0_output": k0_by_index.get(idx, ""),
            "raw_text": stitched,
            "generated_text": continuation,
            "input_prompt": continue_prompt,
            "generate_attempts": attempts_used,
        }
        upsert_jsonl_row(output_path, record)
        n_run += 1

        log_banner("NEW REASONING (continuation only)")
        print(new_reasoning or "(empty)", flush=True)
        log_banner("OUTPUT")
        print(output or "(empty)", flush=True)
        if not output:
            print(
                f"[{dialect} {idx}] {row.get('behavior_id', '')} still has empty "
                "output after retries (re-run this script to try again)",
                flush=True,
            )
        else:
            print(f"[{dialect} {idx}] {row.get('behavior_id', '')} done", flush=True)
    return n_run


def run_dialect(
    args: argparse.Namespace,
    *,
    sampling_params: dict,
    build_continue_prompt,
    parse_generation,
    load_engine,
    model_label: str,
    probe_channel: str,
    classify_script: str,
) -> None:
    from transformers import AutoTokenizer

    probes_path = Path(args.probes)
    if not probes_path.is_file():
        raise SystemExit(f"Probes file not found: {probes_path}")

    records = load_records(probes_path)
    k0_by_index: dict[int, str] = {}
    baseline_path = Path(args.baseline)
    if baseline_path.is_file():
        for row in load_records(baseline_path):
            k0_by_index[row["index"]] = row.get("output", "")

    dialect = args.dialect
    prompt_field = f"prompt_{dialect}"
    probe_field = f"probe_{dialect}"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    print(
        f"[{dialect}] prompt field {prompt_field}, probe field {probe_field}",
        flush=True,
    )
    print(f"[{dialect}] output {output_path}", flush=True)
    pending, n_skip = collect_pending(
        records=records,
        dialect=dialect,
        prompt_field=prompt_field,
        probe_field=probe_field,
        output_path=output_path,
        tokenizer=tokenizer,
        limit=args.limit,
        empty_only=args.empty_only,
        build_continue_prompt=build_continue_prompt,
        model_label=model_label,
        probe_channel=probe_channel,
    )

    if args.dry_run:
        print(
            f"\nDry-run [{dialect}]: {len(pending)} prompt(s) printed; "
            "not loading the GPU",
            flush=True,
        )
        return

    if not pending:
        print(f"[{dialect}] nothing to generate.", flush=True)
        return

    params = dict(sampling_params)
    params["max_new_tokens"] = args.max_new_tokens

    print(f"Loading engine from {args.model_path} ...", flush=True)
    llm, parser = load_engine(args.model_path)
    try:
        n_run = generate_pending(
            llm=llm,
            parser=parser,
            tokenizer=tokenizer,
            sampling_params=params,
            pending=pending,
            dialect=dialect,
            prompt_field=prompt_field,
            probe_field=probe_field,
            output_path=output_path,
            k0_by_index=k0_by_index,
            empty_retries=args.empty_retries,
            model_path=args.model_path,
            model_label=model_label,
            parse_generation=parse_generation,
        )
    finally:
        llm.shutdown()

    print(
        f"\n[{dialect}] wrote {n_run} rows, skipped {n_skip} -> {output_path}",
        flush=True,
    )
    print(
        "Classify with:\n"
        f"  python {classify_script} --input {output_path}",
        flush=True,
    )

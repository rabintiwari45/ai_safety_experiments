#!/usr/bin/env bash
# Score the four Qwen3 StrongREJECT inference JSONLs.
# Run from anywhere. Requires OPENAI_API_KEY.
# LIMIT is the max pending rows scored per file. 313 is the full StrongREJECT set.
# Rows that already have a finite strongreject_score are skipped, so this is safe to re-run.
set -euo pipefail
cd "$(dirname "$0")/../../.."

LIMIT=313

python scripts/strongreject/eval/evaluate.py \
  --input outputs/strongreject/qwen3/probes_reasoning.jsonl \
  --key output \
  --limit "$LIMIT"

python scripts/strongreject/eval/evaluate.py \
  --input outputs/strongreject/qwen3/bypass_reasoning_deepseekv3-03_infer.jsonl \
  --key output \
  --limit "$LIMIT"

python scripts/strongreject/eval/evaluate.py \
  --input outputs/strongreject/qwen3/bypass_reasoning_deepseekv3-03_closed_think_infer.jsonl \
  --key output \
  --limit "$LIMIT"

python scripts/strongreject/eval/evaluate.py \
  --input outputs/strongreject/qwen3/bypass_reasoning_deepseekv3-03_no_think_infer.jsonl \
  --key new_output_suffix \
  --limit "$LIMIT"

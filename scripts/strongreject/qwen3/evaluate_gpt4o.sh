#!/usr/bin/env bash
# Re-score Qwen3 StrongREJECT rows whose judge score is missing or nan.
# Same four inference files as evaluate.sh, with the judge set to openai/gpt-4o.
# Finite strongreject_score rows are skipped. Requires OPENAI_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/../../.."

LIMIT=313

python scripts/strongreject/eval/evaluate.py \
  --input outputs/strongreject/qwen3/probes_reasoning.jsonl \
  --key output \
  --model openai/gpt-4o \
  --limit "$LIMIT"

python scripts/strongreject/eval/evaluate.py \
  --input outputs/strongreject/qwen3/bypass_reasoning_deepseekv3-03_infer.jsonl \
  --key output \
  --model openai/gpt-4o \
  --limit "$LIMIT"

python scripts/strongreject/eval/evaluate.py \
  --input outputs/strongreject/qwen3/bypass_reasoning_deepseekv3-03_closed_think_infer.jsonl \
  --key output \
  --model openai/gpt-4o \
  --limit "$LIMIT"

python scripts/strongreject/eval/evaluate.py \
  --input outputs/strongreject/qwen3/bypass_reasoning_deepseekv3-03_no_think_infer.jsonl \
  --key new_output_suffix \
  --model openai/gpt-4o \
  --limit "$LIMIT"

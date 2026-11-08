#!/usr/bin/env bash
# Classify the four Qwen3-8B contextual-without-context inference JSONLs.
# Run from anywhere. Requires OPENAI_API_KEY.
#
# The classifier sees the behavior and the generation only, not the context.
# Output-prefix rows are scored on new_output_suffix. The other three files
# are scored on output.
#
# Each output is the input name with _classified before .jsonl.
# LIMIT is the max rows scored per file. Default is 100.
# Override without editing: LIMIT=10 bash classify_inferences.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."

LIMIT="${LIMIT:-100}"
DIR=/workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/qwen3
CLASSIFY=scripts/harmbench/contextual_without_context/qwen3/classify.py

python "$CLASSIFY" \
  --input "$DIR/probes_reasoning.jsonl" \
  --output "$DIR/probes_reasoning_classified.jsonl" \
  --key output \
  --limit "$LIMIT"

python "$CLASSIFY" \
  --input "$DIR/bypass_reasoning_deepseekv3-03_infer.jsonl" \
  --output "$DIR/bypass_reasoning_deepseekv3-03_infer_classified.jsonl" \
  --key output \
  --limit "$LIMIT"

python "$CLASSIFY" \
  --input "$DIR/bypass_reasoning_deepseekv3-03_closed_think_infer.jsonl" \
  --output "$DIR/bypass_reasoning_deepseekv3-03_closed_think_infer_classified.jsonl" \
  --key output \
  --limit "$LIMIT"

python "$CLASSIFY" \
  --input "$DIR/bypass_reasoning_deepseekv3-03_no_think_infer.jsonl" \
  --output "$DIR/bypass_reasoning_deepseekv3-03_no_think_infer_classified.jsonl" \
  --key new_output_suffix \
  --limit "$LIMIT"

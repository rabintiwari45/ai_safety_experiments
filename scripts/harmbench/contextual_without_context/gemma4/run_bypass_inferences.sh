#!/usr/bin/env bash
# Three bypass continuations: open thought, output prefix, and closed thought.
# Run from anywhere.
#
# LIMIT is the max pending rows for each command. Default is 100.
# Override without editing: LIMIT=10 bash run_bypass_inferences.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."

LIMIT="${LIMIT:-100}"
DIR=/workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/gemma4
BYPASS="$DIR/bypass_reasoning_deepseekv3-03.jsonl"

python scripts/harmbench/contextual_without_context/gemma4/run_bypass_inference.py \
  --input "$BYPASS" \
  --output "$DIR/bypass_reasoning_deepseekv3-03_infer.jsonl" \
  --limit "$LIMIT"

python scripts/harmbench/contextual_without_context/gemma4/run_bypass_inference_output_prefix.py \
  --input "$BYPASS" \
  --output "$DIR/bypass_reasoning_deepseekv3-03_output_prefix_infer.jsonl" \
  --limit "$LIMIT"

python scripts/harmbench/contextual_without_context/gemma4/run_bypass_inference_closed_think.py \
  --input "$BYPASS" \
  --output "$DIR/bypass_reasoning_deepseekv3-03_closed_think_infer.jsonl" \
  --limit "$LIMIT"

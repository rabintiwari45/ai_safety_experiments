#!/usr/bin/env bash
# Probe inference, then rewrite that reasoning into a bypass trace.
# Run from anywhere. generate_bypass_reasoning.py needs OPENROUTER_API_KEY.
#
# LIMIT is the max pending rows for each command. Default is 100.
# Override without editing: LIMIT=10 bash run_probe_infer_and_bypass.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."

LIMIT="${LIMIT:-100}"
DIR=/workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/gemma4

python scripts/harmbench/contextual_without_context/gemma4/run_inference.py \
  --probes "$DIR/probes.jsonl" \
  --output "$DIR/probes_reasoning.jsonl" \
  --limit "$LIMIT"

python scripts/harmbench/contextual_without_context/gemma4/generate_bypass_reasoning.py \
  --input "$DIR/probes_reasoning.jsonl" \
  --output "$DIR/bypass_reasoning_deepseekv3-03.jsonl" \
  --limit "$LIMIT"

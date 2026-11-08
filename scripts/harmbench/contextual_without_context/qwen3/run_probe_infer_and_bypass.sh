#!/usr/bin/env bash
# Probe inference, then rewrite that reasoning into a bypass trace.
# Run from anywhere. generate_bypass_reasoning.py needs OPENROUTER_API_KEY.
#
# LIMIT is the max pending rows for each command. Default is 1.
# Override without editing: LIMIT=10 bash run_probe_infer_and_bypass.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."

LIMIT="${LIMIT:-100}"

python scripts/harmbench/contextual_without_context/qwen3/run_inference.py \
  --input /workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/qwen3/probes.jsonl \
  --output /workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/qwen3/probes_reasoning.jsonl \
  --limit "$LIMIT"

python scripts/harmbench/contextual_without_context/qwen3/generate_bypass_reasoning.py \
  --input /workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/qwen3/probes_reasoning.jsonl \
  --output /workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/qwen3/bypass_reasoning_deepseekv3-03.jsonl \
  --limit "$LIMIT"

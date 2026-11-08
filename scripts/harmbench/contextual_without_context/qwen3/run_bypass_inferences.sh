#!/usr/bin/env bash
# Three bypass continuations: open think, output prefix, and closed think.
# Run from anywhere.
#
# LIMIT is the max pending rows for each command. Default is 1.
# Override without editing: LIMIT=10 bash run_bypass_inferences.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."

LIMIT="${LIMIT:-100}"

python scripts/harmbench/contextual_without_context/qwen3/run_bypass_inference.py \
  --input /workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/qwen3/bypass_reasoning_deepseekv3-03.jsonl \
  --output /workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/qwen3/bypass_reasoning_deepseekv3-03_infer.jsonl \
  --limit "$LIMIT"

python scripts/harmbench/contextual_without_context/qwen3/run_bypass_inference_output_prefix.py \
  --input /workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/qwen3/bypass_reasoning_deepseekv3-03.jsonl \
  --output /workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/qwen3/bypass_reasoning_deepseekv3-03_no_think_infer.jsonl \
  --limit "$LIMIT"

python scripts/harmbench/contextual_without_context/qwen3/run_bypass_inference_closed_think.py \
  --input /workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/qwen3/bypass_reasoning_deepseekv3-03.jsonl \
  --output /workspace/ai_safety_experiments/outputs/harmbench/contextual_without_context/qwen3/bypass_reasoning_deepseekv3-03_closed_think_infer.jsonl \
  --limit "$LIMIT"

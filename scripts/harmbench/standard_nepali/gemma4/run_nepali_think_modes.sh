#!/usr/bin/env bash
# Gemma 4: Nepali prompt + Nepali bypass, three think modes.
#
# Open think, closed think, and no think. Each mode reads
# bypass_nepali_reasoning_deepseekv3_03.jsonl and writes its own infer JSONL.
#
# From ai_safety_experiments/:
#   bash scripts/harmbench/standard_nepali/gemma4/run_nepali_think_modes.sh
#   LIMIT=1 bash scripts/harmbench/standard_nepali/gemma4/run_nepali_think_modes.sh
#
# Default limit is 1 pending row per mode. Finished rows are skipped.

set -euo pipefail

ROOT="/workspace/ai_safety_experiments"
MODEL_DIR="scripts/harmbench/standard_nepali/gemma4"

if [[ $# -ge 1 && "$1" != -* ]]; then
  LIMIT="$1"
  shift
fi
if [[ -z "${LIMIT:-}" ]]; then
  LIMIT=200
fi

cd "${ROOT}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET=1

scripts=(
  run_bypass_inference.py
  run_bypass_inference_closed_think.py
  run_bypass_inference_output_prefix.py
  
)

for script in "${scripts[@]}"; do
  echo "=== ${script} ==="
  python3 "${MODEL_DIR}/${script}" --limit "${LIMIT}" "$@"
done

#!/usr/bin/env bash
# Qwen3-4B: dialect prompt + dialect bypass, both dialects, three think modes.
#
# Open think, closed think, and no think. Each mode reads
# bypass_reasoning_deepseekv3-03_<dialect>_dialect.jsonl and writes its own
# infer JSONL next to it.
#
# From ai_safety_experiments/:
#   bash scripts/multivalue_dialect/qwen3-4b/run_dialect_think_modes.sh
#   LIMIT=1 bash scripts/multivalue_dialect/qwen3-4b/run_dialect_think_modes.sh
#   bash scripts/multivalue_dialect/qwen3-4b/run_dialect_think_modes.sh --dry-run --limit 1
#
# Omit LIMIT to run every pending row. Rows with a non-empty output are skipped.

set -euo pipefail

ROOT="/workspace/ai_safety_experiments"
MODEL_DIR="scripts/multivalue_dialect/qwen3-4b"

extra=()
if [[ $# -ge 1 && "$1" != -* ]]; then
  LIMIT="$1"
  shift
fi
if [[ -n "${LIMIT:-}" ]]; then
  extra+=(--limit "${LIMIT}")
fi
extra+=("$@")

cd "${ROOT}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET=1

dialects=(
  african_american_vernacular
  colloquial_singapore
)
scripts=(
  run_bypass_inference.py
  run_bypass_inference_closed_think.py
  run_bypass_inference_output_prefix.py
)

for dialect in "${dialects[@]}"; do
  for script in "${scripts[@]}"; do
    echo "=== ${script} --dialect ${dialect} ==="
    python3 "${MODEL_DIR}/${script}" --dialect "${dialect}" "${extra[@]}"
  done
done

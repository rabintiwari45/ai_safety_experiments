#!/usr/bin/env bash
# Transform bypass_reasoning for both dialects on Qwen3-8B, Qwen3-4B, and Gemma.
#
# From ai_safety_experiments/:
#   bash scripts/multivalue_dialect/transform_bypass_dialects.sh
#   LIMIT=10 bash scripts/multivalue_dialect/transform_bypass_dialects.sh
#   bash scripts/multivalue_dialect/transform_bypass_dialects.sh 10
#
# Omit the limit to transform every row:
#   LIMIT= bash scripts/multivalue_dialect/transform_bypass_dialects.sh
#
# Each input is left unchanged. Output is <input stem>_dialect.jsonl with
# bypass_reasoning_african_american_vernacular or
# bypass_reasoning_colloquial_singapore added.

set -euo pipefail

ROOT="/workspace/ai_safety_experiments"
BASE="${ROOT}/outputs/multivalue_dialect/harmbench/standard"

# First argument wins, then LIMIT, then 1.
if [[ $# -ge 1 ]]; then
  LIMIT="$1"
elif [[ -v LIMIT ]]; then
  LIMIT="${LIMIT}"
else
  LIMIT="200"
fi

cd "${ROOT}"

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

models=(qwen3-8b qwen3-4b gemma4)
dialects=(
  african_american_vernacular
  colloquial_singapore
)

for model in "${models[@]}"; do
  for dialect in "${dialects[@]}"; do
    input="${BASE}/${model}/bypass_reasoning_deepseekv3-03_${dialect}.jsonl"
    python3 scripts/multivalue_dialect/transform_bypass_reasoning.py \
      --input "${input}" \
      --dialect "${dialect}" \
      "${limit_args[@]}"
  done
done

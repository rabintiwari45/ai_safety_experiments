#!/usr/bin/env bash
# Rewrite Gemma 4 dialect reasoning with DeepSeek, both dialects.
#
# From ai_safety_experiments/:
#   bash scripts/multivalue_dialect/gemma4/run_dialect_bypass.sh
#   LIMIT=10 bash scripts/multivalue_dialect/gemma4/run_dialect_bypass.sh
#   bash scripts/multivalue_dialect/gemma4/run_dialect_bypass.sh 10
#
# Omit the limit to run every pending row:
#   LIMIT= bash scripts/multivalue_dialect/gemma4/run_dialect_bypass.sh
#
# Writes:
#   outputs/multivalue_dialect/harmbench/standard/gemma4/bypass_reasoning_deepseekv3-03_african_american_vernacular.jsonl
#   outputs/multivalue_dialect/harmbench/standard/gemma4/bypass_reasoning_deepseekv3-03_colloquial_singapore.jsonl

set -euo pipefail

ROOT="/workspace/ai_safety_experiments"

# First argument wins, then LIMIT, then 200.
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

python3 scripts/multivalue_dialect/gemma4/generate_bypass_reasoning.py \
  --dialect african_american_vernacular \
  "${limit_args[@]}"

python3 scripts/multivalue_dialect/gemma4/generate_bypass_reasoning.py \
  --dialect colloquial_singapore \
  "${limit_args[@]}"

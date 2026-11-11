#!/usr/bin/env bash
# Run Gemma 4 E4B on both dialect probes.
#
# From ai_safety_experiments/:
#   bash scripts/multivalue_dialect/gemma4/run_dialect_probes.sh
#   LIMIT=10 bash scripts/multivalue_dialect/gemma4/run_dialect_probes.sh
#   bash scripts/multivalue_dialect/gemma4/run_dialect_probes.sh 10
#
# Omit the limit to run every pending row:
#   LIMIT= bash scripts/multivalue_dialect/gemma4/run_dialect_probes.sh

set -euo pipefail

ROOT="/workspace/ai_safety_experiments"
PROBES="${ROOT}/outputs/multivalue_dialect/harmbench/standard/gemma4/probes_dialects.jsonl"
OUT_DIR="${ROOT}/outputs/multivalue_dialect/harmbench/standard/gemma4"

# First argument wins, then LIMIT, then 200.
if [[ $# -ge 1 ]]; then
  LIMIT="$1"
elif [[ -v LIMIT ]]; then
  LIMIT="${LIMIT}"
else
  LIMIT="1"
fi

cd "${ROOT}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET=1

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

python3 scripts/multivalue_dialect/gemma4/run_african_american_vernacular.py \
  --probes "${PROBES}" \
  --output "${OUT_DIR}/probes_dialects_reasoning_african_american.jsonl" \
  "${limit_args[@]}"

python3 scripts/multivalue_dialect/gemma4/run_colloquial_singapore.py \
  --probes "${PROBES}" \
  --output "${OUT_DIR}/probes_dialects_reasoning_singapore.jsonl" \
  "${limit_args[@]}"

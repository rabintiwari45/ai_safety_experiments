#!/usr/bin/env bash
# Run Gemma 4 E4B on both StrongREJECT dialect probes.
#
# User message: prompt_<dialect>
# Open thought prefix: probe_<dialect>
#
# From ai_safety_experiments/ (default limit is 313):
#   bash scripts/multivalue_dialect/strongreject/gemma4/run_dialect_probes.sh
#   LIMIT=1 bash scripts/multivalue_dialect/strongreject/gemma4/run_dialect_probes.sh
#   bash scripts/multivalue_dialect/strongreject/gemma4/run_dialect_probes.sh 10
#
# Omit the limit to run every pending row:
#   LIMIT= bash scripts/multivalue_dialect/strongreject/gemma4/run_dialect_probes.sh

set -euo pipefail

ROOT="/workspace/ai_safety_experiments"

# First argument wins, then LIMIT, then 313.
if [[ $# -ge 1 ]]; then
  LIMIT="$1"
elif [[ -v LIMIT ]]; then
  LIMIT="${LIMIT}"
else
  LIMIT="313"
fi

cd "${ROOT}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_DISABLE_XET=1

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

python3 scripts/multivalue_dialect/strongreject/gemma4/run_inference.py \
  --dialect african_american_vernacular \
  "${limit_args[@]}"

python3 scripts/multivalue_dialect/strongreject/gemma4/run_inference.py \
  --dialect colloquial_singapore \
  "${limit_args[@]}"

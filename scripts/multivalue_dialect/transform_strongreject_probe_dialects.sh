#!/usr/bin/env bash
# Transform StrongREJECT prompt and probe into both VALUE dialects.
#
# Uses the same dialect classes as the HarmBench standard probes transform:
#   AfricanAmericanVernacular  -> prompt_african_american_vernacular, probe_african_american_vernacular
#   ColloquialSingaporeDialect -> prompt_colloquial_singapore, probe_colloquial_singapore
#
# Input (313 rows) is left unchanged:
#   outputs/multivalue_dialect/strongreject/qwen3-8b/probes.jsonl
# Output:
#   outputs/multivalue_dialect/strongreject/qwen3-8b/probes_dialects.jsonl
#
# From ai_safety_experiments/ (default limit is 313):
#   bash scripts/multivalue_dialect/transform_strongreject_probe_dialects.sh
#   LIMIT=10 bash scripts/multivalue_dialect/transform_strongreject_probe_dialects.sh
#   bash scripts/multivalue_dialect/transform_strongreject_probe_dialects.sh 10
#
# Omit the limit to transform every row (313):
#   LIMIT= bash scripts/multivalue_dialect/transform_strongreject_probe_dialects.sh

set -euo pipefail

ROOT="/workspace/ai_safety_experiments"
INPUT="${ROOT}/outputs/multivalue_dialect/strongreject/qwen3-8b/probes.jsonl"
OUTPUT="${ROOT}/outputs/multivalue_dialect/strongreject/qwen3-8b/probes_dialects.jsonl"

# First argument wins, then LIMIT, then 313.
if [[ $# -ge 1 ]]; then
  LIMIT="$1"
elif [[ -v LIMIT ]]; then
  LIMIT="${LIMIT}"
else
  LIMIT="313"
fi

cd "${ROOT}"

limit_args=()
if [[ -n "${LIMIT}" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

python3 scripts/misc/multivalue/transform_probe_dialects.py \
  --input "${INPUT}" \
  --output "${OUTPUT}" \
  --dialects AfricanAmericanVernacular ColloquialSingaporeDialect \
  "${limit_args[@]}"

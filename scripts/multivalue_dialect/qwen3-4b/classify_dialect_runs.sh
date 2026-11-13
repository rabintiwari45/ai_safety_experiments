#!/usr/bin/env bash
# Classify the eight Qwen3-4B dialect files.
#
# Probe reasoning, open think, and closed think are scored on output.
# No-think is scored on new_output_suffix, the text the model added after
# the prefilled bypass.
# Each classified file is saved beside its input as <stem>_classified.jsonl.
#
# From ai_safety_experiments/:
#   export OPENAI_API_KEY=...
#   bash scripts/multivalue_dialect/qwen3-4b/classify_dialect_runs.sh
#   LIMIT=200 bash scripts/multivalue_dialect/qwen3-4b/classify_dialect_runs.sh
#
# Default limit is 200 rows per file. Rows that already have a yes/no label
# are skipped.

set -euo pipefail

ROOT="/workspace/ai_safety_experiments"
cd "${ROOT}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Set OPENAI_API_KEY before classifying." >&2
  exit 1
fi

CLASSIFY="scripts/multivalue_dialect/qwen3-4b/classify.py"
DIR="outputs/multivalue_dialect/harmbench/standard/qwen3-4b"
LIMIT="${LIMIT:-200}"

classify_file() {
  local input="$1"
  local key="$2"
  if [[ ! -f "${input}" ]]; then
    echo "Missing ${input}" >&2
    exit 1
  fi
  echo "=== $(basename "${input}") (${key}) ==="
  python3 "${CLASSIFY}" \
    --input "${input}" \
    --output "${input%.jsonl}_classified.jsonl" \
    --key "${key}" \
    --limit "${LIMIT}"
}

classify_file "${DIR}/probes_dialects_reasoning_african_american.jsonl" output
classify_file "${DIR}/probes_dialects_reasoning_singapore.jsonl" output

for dialect in african_american_vernacular colloquial_singapore; do
  base="${DIR}/bypass_reasoning_deepseekv3-03_${dialect}_dialect"
  classify_file "${base}_infer.jsonl" output
  classify_file "${base}_closed_think_infer.jsonl" output
  classify_file "${base}_no_think_infer.jsonl" new_output_suffix
done

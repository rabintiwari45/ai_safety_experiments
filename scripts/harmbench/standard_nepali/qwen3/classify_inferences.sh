#!/usr/bin/env bash
# Classify the Qwen3-8B Nepali runs from their English translations.
#
# Behavior is prompt_en. Open think, closed think, and probes are scored
# on output. No think is scored on new_output_suffix.
#
# Each classified file is saved beside its English input as
# <stem>_classified.jsonl. Rows that already have a yes/no label are skipped.
#
# From ai_safety_experiments/:
#   export OPENAI_API_KEY=...
#   bash scripts/harmbench/standard_nepali/qwen3/classify_inferences.sh
#   LIMIT=200 bash scripts/harmbench/standard_nepali/qwen3/classify_inferences.sh
#   bash scripts/harmbench/standard_nepali/qwen3/classify_inferences.sh --dry-run

set -euo pipefail

ROOT="/workspace/ai_safety_experiments"
OUT_DIR="outputs/harmbench/standard_nepali/qwen3-8b"
PY="scripts/harmbench/standard_nepali/qwen3/classify.py"

if [[ $# -ge 1 && "$1" != -* ]]; then
  LIMIT="$1"
  shift
fi
if [[ -z "${LIMIT:-}" ]]; then
  LIMIT=200
fi

cd "${ROOT}"

dry=0
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" || "${arg}" == "--print-prompt" ]]; then
    dry=1
  fi
done
if [[ "${dry}" -eq 0 && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Set OPENAI_API_KEY before classifying." >&2
  exit 1
fi

# name, English generation field
jobs=(
  "bypass_nepali_reasoning_deepseekv3_03_infer_en.jsonl output"
  "bypass_nepali_reasoning_deepseekv3_03_closed_think_infer_en.jsonl output"
  "bypass_nepali_reasoning_deepseekv3_03_no_think_infer_en.jsonl new_output_suffix"
  "probes_nepali_reasoning_en.jsonl output"
)

for job in "${jobs[@]}"; do
  name="${job%% *}"
  field="${job##* }"
  src="${OUT_DIR}/${name}"
  dst="${OUT_DIR}/${name%.jsonl}_classified.jsonl"
  echo "=== ${name} (prompt_en, ${field}) ==="
  python3 "${PY}" \
    --input "${src}" \
    --output "${dst}" \
    --key "${field}" \
    --limit "${LIMIT}" \
    "$@"
done

#!/usr/bin/env bash
# Translate every Gemma 4 Nepali infer JSONL to English.
#
# Each file is read as --input and written to a new *_en.jsonl. The source
# files are not modified. Finished rows in an existing *_en.jsonl are skipped.
#
# Open think, closed think, and probes_nepali_reasoning translate `output`.
# No think translates `new_output_suffix`.
#
# Default limit is 200 pending rows per file.
#
# From ai_safety_experiments/:
#   export OPENROUTER_API_KEY=...
#   bash scripts/harmbench/standard_nepali/gemma4/translate_infer_to_english.sh
#   LIMIT=200 bash scripts/harmbench/standard_nepali/gemma4/translate_infer_to_english.sh
#   bash scripts/harmbench/standard_nepali/gemma4/translate_infer_to_english.sh --dry-run

set -euo pipefail

ROOT="/workspace/ai_safety_experiments"
OUT_DIR="outputs/harmbench/standard_nepali/gemma4"
PY="scripts/harmbench/standard_nepali/gemma4/translate_infer_to_english.py"

if [[ $# -ge 1 && "$1" != -* ]]; then
  LIMIT="$1"
  shift
fi
if [[ -z "${LIMIT:-}" ]]; then
  LIMIT=200
fi

cd "${ROOT}"

# name, field to translate
jobs=(
  "bypass_nepali_reasoning_deepseekv3_03_infer.jsonl output"
  "bypass_nepali_reasoning_deepseekv3_03_closed_think_infer.jsonl output"
  "bypass_nepali_reasoning_deepseekv3_03_no_think_infer.jsonl new_output_suffix"
  "probes_nepali_reasoning.jsonl output"
)

extra=()
if [[ -n "${LIMIT:-}" ]]; then
  extra+=(--limit "${LIMIT}")
fi

for job in "${jobs[@]}"; do
  name="${job%% *}"
  field="${job##* }"
  src="${OUT_DIR}/${name}"
  dst="${OUT_DIR}/${name%.jsonl}_en.jsonl"
  echo "=== ${name} (${field}) ==="
  python3 "${PY}" --input "${src}" --output "${dst}" --field "${field}" "${extra[@]}" "$@"
done

#!/usr/bin/env bash
# Print the mean strongreject_score for each scored Qwen3 file.
# Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/../../.."

PY=python
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python3
fi

for stem in \
  probes_reasoning \
  bypass_reasoning_deepseekv3-03_infer \
  bypass_reasoning_deepseekv3-03_closed_think_infer \
  bypass_reasoning_deepseekv3-03_no_think_infer
do
  file="outputs/strongreject/qwen3/${stem}_strongreject_rubric.jsonl"
  echo
  echo "[qwen3] ${file}"
  if [[ ! -f "$file" ]]; then
    echo "missing"
    continue
  fi
  "$PY" scripts/strongreject/eval/mean_score.py --input "$file"
done
echo

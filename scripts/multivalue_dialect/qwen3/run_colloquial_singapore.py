"""
Run Qwen3-8B on Colloquial Singapore dialect probes.

User message: prompt_colloquial_singapore
Open <think> prefix: probe_colloquial_singapore

Writes outputs/multivalue_dialect/harmbench/standard/qwen3-8b/probes_infer_colloquial_singapore.jsonl

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/qwen3/run_colloquial_singapore.py --dry-run --limit 1
    python scripts/multivalue_dialect/qwen3/run_colloquial_singapore.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import main_for  # noqa: E402

if __name__ == "__main__":
    main_for("colloquial_singapore")

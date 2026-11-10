"""
Run Gemma 4 E4B (google/gemma-4-E4B-it) on Colloquial Singapore dialect probes.

User message: prompt_colloquial_singapore
Open thought prefix: probe_colloquial_singapore

Writes outputs/multivalue_dialect/harmbench/standard/gemma4/probes_infer_colloquial_singapore.jsonl

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/gemma4/run_colloquial_singapore.py --dry-run --limit 1
    python scripts/multivalue_dialect/gemma4/run_colloquial_singapore.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import main_for  # noqa: E402

if __name__ == "__main__":
    main_for("colloquial_singapore")

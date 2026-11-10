"""
Run Gemma 4 E4B (google/gemma-4-E4B-it) on African American Vernacular probes.

User message: prompt_african_american_vernacular
Open thought prefix: probe_african_american_vernacular

Writes outputs/multivalue_dialect/harmbench/standard/gemma4/probes_infer_african_american_vernacular.jsonl

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/gemma4/run_african_american_vernacular.py --dry-run --limit 1
    python scripts/multivalue_dialect/gemma4/run_african_american_vernacular.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import main_for  # noqa: E402

if __name__ == "__main__":
    main_for("african_american_vernacular")

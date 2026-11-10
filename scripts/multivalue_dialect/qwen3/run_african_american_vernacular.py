"""
Run Qwen3-8B on African American Vernacular dialect probes.

User message: prompt_african_american_vernacular
Open <think> prefix: probe_african_american_vernacular

Writes outputs/multivalue_dialect/harmbench/standard/qwen3-8b/probes_infer_african_american_vernacular.jsonl

Usage (from ai_safety_experiments/):
    python scripts/multivalue_dialect/qwen3/run_african_american_vernacular.py --dry-run --limit 1
    python scripts/multivalue_dialect/qwen3/run_african_american_vernacular.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime import main_for  # noqa: E402

if __name__ == "__main__":
    main_for("african_american_vernacular")

"""
Transform bypass_reasoning (or --source-key) with AfricanAmericanVernacular.

Urban African American Vernacular English in VALUE (multivalue).

Usage (from ai_safety_experiments/):
    python scripts/misc/multivalue/transform_african_american_vernacular.py --dry-run --limit 1
    python scripts/misc/multivalue/transform_african_american_vernacular.py
    python scripts/misc/multivalue/transform_african_american_vernacular.py --overwrite
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multivalue.Dialects import AfricanAmericanVernacular

from dialect_transform_common import main_preset

DIALECT_CLASS_NAME = "AfricanAmericanVernacular"
DEFAULT_OUTPUT_KEY = "bypass_reasoning_african_american_vernacular"
OUTPUT_FILE_TAG = "african_american_vernacular"


def main() -> None:
    main_preset(
        description="VALUE transform using AfricanAmericanVernacular",
        dialect_cls=AfricanAmericanVernacular,
        dialect_class_name=DIALECT_CLASS_NAME,
        output_file_tag=OUTPUT_FILE_TAG,
        default_output_key=DEFAULT_OUTPUT_KEY,
    )


if __name__ == "__main__":
    main()

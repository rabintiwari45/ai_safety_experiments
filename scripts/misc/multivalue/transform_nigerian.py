"""
Transform bypass_reasoning (or --source-key) with NigerianDialect.

Usage (from ai_safety_experiments/):
    python scripts/misc/multivalue/transform_nigerian.py --dry-run --limit 1
    python scripts/misc/multivalue/transform_nigerian.py
    python scripts/misc/multivalue/transform_nigerian.py --overwrite
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from multivalue.Dialects import NigerianDialect

from dialect_transform_common import main_preset

DIALECT_CLASS_NAME = "NigerianDialect"
DEFAULT_OUTPUT_KEY = "bypass_reasoning_nigerian"
OUTPUT_FILE_TAG = "nigerian"


def main() -> None:
    main_preset(
        description="VALUE transform using NigerianDialect",
        dialect_cls=NigerianDialect,
        dialect_class_name=DIALECT_CLASS_NAME,
        output_file_tag=OUTPUT_FILE_TAG,
        default_output_key=DEFAULT_OUTPUT_KEY,
    )


if __name__ == "__main__":
    main()

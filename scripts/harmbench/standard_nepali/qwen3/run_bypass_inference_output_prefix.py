"""Qwen3-8B no-think run on the Nepali prompt and Nepali bypass."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qwen_bypass_runtime import main_for  # noqa: E402

if __name__ == "__main__":
    main_for("no_think", "qwen3-8b")

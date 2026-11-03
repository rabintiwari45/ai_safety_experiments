"""Paths, model id, and StrongREJECT loader for Qwen3-8B."""

from __future__ import annotations

import csv
import io
import sys
from pathlib import Path
from urllib.request import urlopen

# scripts/strongreject/qwen3/<file>.py -> scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "misc"))

from common import OUTPUT_DIR, PROJECT_ROOT  # noqa: E402

# Same thinking model as the HarmBench Qwen3-8B runs. enable_thinking and <think> still apply.
MODEL_PATH = "Qwen/Qwen3-8B"

STRONGREJECT_CSV_URL = (
    "https://raw.githubusercontent.com/alexandrasouly/strongreject/"
    "main/strongreject_dataset/strongreject_dataset.csv"
)
EXPECTED_ROWS = 313

QWEN3_DIR = OUTPUT_DIR / "strongreject" / "qwen3"
DATA_CSV = PROJECT_ROOT / "data" / "strongreject" / "strongreject_dataset.csv"

BASELINE_JSONL = QWEN3_DIR / "baseline.jsonl"
PROBES = QWEN3_DIR / "probes.jsonl"
PROBE_INFER = QWEN3_DIR / "probes_reasoning.jsonl"
BYPASS = QWEN3_DIR / "bypass_reasoning_deepseekv3-03.jsonl"
BYPASS_INFER = QWEN3_DIR / "bypass_reasoning_deepseekv3-03_infer.jsonl"
BYPASS_INFER_CLASSIFIED = (
    QWEN3_DIR / "bypass_reasoning_deepseekv3-03_infer_classified.jsonl"
)
BYPASS_OUTPUT_PREFIX_INFER = (
    QWEN3_DIR / "bypass_reasoning_deepseekv3-03_no_think_infer.jsonl"
)
BYPASS_OUTPUT_PREFIX_CLASSIFIED = (
    QWEN3_DIR / "bypass_reasoning_deepseekv3-03_no_think_infer_classified.jsonl"
)


def load_strongreject_behaviors(csv_path: Path | None = None) -> list[dict]:
    """Load the public StrongREJECT forbidden-prompt CSV (313 rows).

    Columns used: category, source, forbidden_prompt.
    Each row is shaped like a HarmBench behavior so the Qwen3 scripts can share fields:
    index, behavior_id, category, source, prompt.
    """
    path = Path(csv_path) if csv_path else DATA_CSV
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        print(f"Downloading StrongREJECT from {STRONGREJECT_CSV_URL} ...")
        with urlopen(STRONGREJECT_CSV_URL, timeout=60) as resp:
            text = resp.read().decode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Cached StrongREJECT CSV at {path}")

    rows: list[dict] = []
    for row in csv.DictReader(io.StringIO(text)):
        prompt = (row.get("forbidden_prompt") or "").strip()
        if not prompt:
            continue
        idx = len(rows)
        source = (row.get("source") or "").strip()
        rows.append(
            {
                "index": idx,
                "prompt": prompt,
                "category": (row.get("category") or "").strip(),
                "source": source,
                "behavior_id": f"strongreject_{idx:03d}",
            }
        )
    if len(rows) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} StrongREJECT prompts, got {len(rows)} from {path}"
        )
    return rows

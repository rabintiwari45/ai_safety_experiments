"""Shared paths and JSONL helpers for standard and dynamic HarmBench scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path


def env_openai_api_key() -> str:
    """OpenAI API key from OPENAI_API_KEY only (never read from source files)."""
    return os.environ.get("OPENAI_API_KEY", "")

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STANDARD_DIR = OUTPUT_DIR / "standard"
DYNAMIC_DIR = OUTPUT_DIR / "dynamic"
HIJACK_REASONING_DIR = OUTPUT_DIR / "hijack_reasoninig"
INTERVENTIONS_DIR = PROJECT_ROOT / "interventions"

MODEL_PATH = "Qwen/Qwen3-8B"

THINKING_SAMPLING_PARAMS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "max_new_tokens": 2048,
    "skip_special_tokens": False,
}

BASELINE_JSON = STANDARD_DIR / "harmbench_standard_qwen3_8b.json"
BASELINE_JSONL = STANDARD_DIR / "harmbench_standard_qwen3_8b.jsonl"
K1_JSONL = STANDARD_DIR / "harmbench_standard_qwen3_8b_k1_B-I1.jsonl"
K2_JSONL = STANDARD_DIR / "harmbench_standard_qwen3_8b_k2_B-I1.jsonl"
K3_JSONL = STANDARD_DIR / "harmbench_standard_qwen3_8b_k3_B-I1.jsonl"
EXPERIMENT1 = INTERVENTIONS_DIR / "experiment1.json"

DYNAMIC_PREFIXES = DYNAMIC_DIR / "dynamic_think_prefixes.jsonl"
DYNAMIC_PREFIXES_ANSWER = DYNAMIC_DIR / "dynamic_think_prefixes_answer.jsonl"
DYNAMIC_JSONL = DYNAMIC_DIR / "harmbench_standard_qwen3_8b_dynamic.jsonl"
DYNAMIC_JSON = DYNAMIC_DIR / "harmbench_standard_qwen3_8b_dynamic.json"
DYNAMIC_ANSWER_JSONL = DYNAMIC_DIR / "harmbench_standard_qwen3_8b_dynamic_answer.jsonl"
DYNAMIC_K1_PROBES = DYNAMIC_DIR / "dynamic_k1_probes.jsonl"
DYNAMIC_K1_JSONL = DYNAMIC_DIR / "harmbench_standard_qwen3_8b_dynamic_k1.jsonl"
DYNAMIC_K2_PROBES = DYNAMIC_DIR / "dynamic_k2_probes.jsonl"
DYNAMIC_K2_JSONL = DYNAMIC_DIR / "harmbench_standard_qwen3_8b_dynamic_k2.jsonl"
DYNAMIC_K3_PROBES = DYNAMIC_DIR / "dynamic_k3_probes.jsonl"
DYNAMIC_K3_JSONL = DYNAMIC_DIR / "harmbench_standard_qwen3_8b_dynamic_k3.jsonl"

TRANSLATION_DIR = OUTPUT_DIR / "translation"
TRANSLATION_JSONL = TRANSLATION_DIR / "dynamic_nepali.jsonl"
TRANSLATION_RUN_JSONL = (
    TRANSLATION_DIR / "harmbench_standard_qwen3_8b_dynamic_nepali.jsonl"
)
TRANSLATION_NOPROBE_JSONL = (
    TRANSLATION_DIR / "harmbench_standard_qwen3_8b_dynamic_nepali_noprobe.jsonl"
)
TRANSLATION_PROBE_JSONL = (
    TRANSLATION_DIR / "harmbench_standard_qwen3_8b_dynamic_nepali_probe.jsonl"
)
TRANSLATION_OUTPUT_EN_JSONL = (
    TRANSLATION_DIR / "harmbench_standard_qwen3_8b_dynamic_nepali_output_en.jsonl"
)
TRANSLATION_OUTPUT_EN_CLASSIFIED_EN = (
    TRANSLATION_DIR
    / "harmbench_standard_qwen3_8b_dynamic_nepali_output_en_classified_en.jsonl"
)
TRANSLATION_K1_PROBES = TRANSLATION_DIR / "dynamic_nepali_k1_probes.jsonl"
TRANSLATION_K1_PROBES_NE = TRANSLATION_DIR / "dynamic_nepali_k1_probes_ne.jsonl"
TRANSLATION_K1_JSONL = (
    TRANSLATION_DIR / "harmbench_standard_qwen3_8b_dynamic_nepali_k1.jsonl"
)
TRANSLATION_K1_NE_JSONL = (
    TRANSLATION_DIR / "harmbench_standard_qwen3_8b_dynamic_nepali_k1_ne.jsonl"
)
TRANSLATION_K1_NE_OUTPUT_EN = (
    TRANSLATION_DIR
    / "harmbench_standard_qwen3_8b_dynamic_nepali_k1_ne_output_en.jsonl"
)
TRANSLATION_K1_NE_GENERATED_EN = (
    TRANSLATION_DIR
    / "harmbench_standard_qwen3_8b_dynamic_nepali_k1_ne_generated_en.jsonl"
)

HIJACK_REASONING_JSONL = HIJACK_REASONING_DIR / "hijack_reasoninig.jsonl"
HIJACK_REASONING_PROBES = HIJACK_REASONING_DIR / "hijack_reasoninig_probes.jsonl"
HIJACK_REASONING_BYPASS = HIJACK_REASONING_DIR / "hijack_reasoninig_bypass.jsonl"
HIJACK_REASONING_BYPASS_V2_DIR = HIJACK_REASONING_DIR / "bypass_v2"
HIJACK_REASONING_BYPASS_V2 = (
    HIJACK_REASONING_BYPASS_V2_DIR / "hijack_reasoninig_bypass_v2.jsonl"
)
HIJACK_REASONING_PROBES_BYPASS = (
    HIJACK_REASONING_DIR / "hijack_reasoninig_probes_bypass.jsonl"
)
HIJACK_REASONING_BYPASS_RUN = (
    HIJACK_REASONING_DIR / "hijack_reasoninig_bypass_run.jsonl"
)
HIJACK_REASONING_BYPASS_RUN_CLASSIFIED = (
    HIJACK_REASONING_DIR / "hijack_reasoninig_bypass_run_classified.jsonl"
)
HIJACK_CLASSIFIER_MODEL = "gpt-5.6-luna"
HIJACK_BYPASS_DEEPSEEKV3 = (
    HIJACK_REASONING_DIR / "hijack_reasoninig_bypass_deepseekv3.jsonl"
)
HIJACK_BYPASS_DEEPSEEKV3_NE = (
    HIJACK_REASONING_DIR / "hijack_reasoninig_bypass_deepseekv3_ne.jsonl"
)
HIJACK_BYPASS_DEEPSEEKV3_NE_INFER = (
    HIJACK_REASONING_DIR / "hijack_reasoninig_bypass_deepseekv3_ne_infer.jsonl"
)
MY_BYPASS_NE_INFER = HIJACK_REASONING_DIR / "my_bypass_ne_infer.jsonl"


def load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return data
    rows: list[dict] = []
    try:
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if rows:
            return rows
    except json.JSONDecodeError:
        rows = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        rows.append(obj)
        idx = end
    return rows


def load_completed_ids(
    output_path: Path,
    *,
    require_output: bool = False,
    output_field: str = "output",
) -> set[int]:
    """Indices present in output JSONL; last row per index wins if duplicated."""
    if not output_path.exists():
        return set()
    by_index: dict[int, dict] = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = row.get("index")
            if idx is not None:
                by_index[int(idx)] = row
    if require_output:
        return {
            idx
            for idx, row in by_index.items()
            if (row.get(output_field) or "").strip()
        }
    return set(by_index.keys())


def load_infer_resume_state(
    output_path: Path,
    output_field: str = "output",
) -> tuple[set[int], set[int]]:
    """
    For inference resume: (indices with non-empty output_field,
    indices in file but output_field empty/missing).
    """
    if not output_path.exists():
        return set(), set()
    by_index: dict[int, dict] = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = row.get("index")
            if idx is not None:
                by_index[int(idx)] = row
    completed = {
        idx
        for idx, row in by_index.items()
        if (row.get(output_field) or "").strip()
    }
    empty = set(by_index.keys()) - completed
    return completed, empty


def upsert_jsonl_row(output_path: Path, record: dict, key: str = "index") -> None:
    """Merge one record into JSONL by key (rewrite file, sorted by key)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    by_key: dict[int, dict] = {}
    if output_path.exists():
        for row in load_records(output_path):
            k = row.get(key)
            if k is not None:
                by_key[int(k)] = row
    k = record.get(key)
    if k is None:
        raise ValueError(f"record missing {key!r}")
    by_key[int(k)] = record
    with output_path.open("w", encoding="utf-8") as f:
        for k in sorted(by_key.keys()):
            f.write(json.dumps(by_key[k], ensure_ascii=False) + "\n")


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text or "", add_special_tokens=False))

"""Shared paths and JSONL helpers for standard and dynamic HarmBench scripts."""

from __future__ import annotations

import json
import os
from pathlib import Path


def env_openai_api_key() -> str:
    """OpenAI API key from OPENAI_API_KEY only (never read from source files)."""
    return os.environ.get("OPENAI_API_KEY", "")

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_DIR.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
STANDARD_DIR = OUTPUT_DIR / "standard"
DYNAMIC_DIR = OUTPUT_DIR / "dynamic"
HIJACK_REASONING_DIR = OUTPUT_DIR / "hijack_reasoninig"
INTERVENTIONS_DIR = PROJECT_ROOT / "interventions"

MODEL_PATH = "Qwen/Qwen3-8B"

_QWEN3_8B_SHARD_NAMES = tuple(
    f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)
)


def _parse_env_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].strip()
    if "=" not in line:
        return None
    key, _, raw = line.partition("=")
    key = key.strip()
    val = raw.strip().strip('"').strip("'")
    if not key:
        return None
    return key, val


def load_workspace_env_file() -> None:
    """Apply ${WORKSPACE}/.env; HF_* keys from the file override /etc/environment."""
    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    env_path = workspace / ".env"
    if not env_path.is_file():
        return
    hf_keys = frozenset({"HF_HOME", "HF_HUB_DISABLE_XET", "HF_TOKEN"})
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(line)
        if parsed is None:
            continue
        key, val = parsed
        if key in hf_keys:
            os.environ[key] = val
        else:
            os.environ.setdefault(key, val)


def _hf_snapshot_dir(hf_home: Path, model_id: str) -> Path | None:
    repo = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    ref_main = repo / "refs" / "main"
    if ref_main.is_file():
        snap = repo / "snapshots" / ref_main.read_text(encoding="utf-8").strip()
        if snap.is_dir():
            return snap
    snaps = repo / "snapshots"
    if not snaps.is_dir():
        return None
    candidates = [p for p in snaps.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def hf_hub_cache_complete(hf_home: Path, model_id: str) -> bool:
    snap = _hf_snapshot_dir(hf_home, model_id)
    if snap is None:
        return False
    if model_id == MODEL_PATH:
        return all(os.path.lexists(snap / name) for name in _QWEN3_8B_SHARD_NAMES)
    index = snap / "model.safetensors.index.json"
    if not index.is_file():
        return (snap / "model.safetensors").is_file()
    weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map", {})
    shard_names = {Path(v).name for v in weight_map.values()}
    return bool(shard_names) and all(
        os.path.lexists(snap / name) for name in shard_names
    )


def configure_model_hub_env(model_id: str = MODEL_PATH) -> str:
    """
    Load workspace .env and point HF_HOME at a complete local cache when possible.

    Must run before importing transformers or sglang (they read HF_HOME at load time).
    """
    load_workspace_env_file()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    workspace = Path(os.environ.get("WORKSPACE", "/workspace"))
    candidates: list[Path] = []
    seen: set[str] = set()

    def add_candidate(path: Path) -> None:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    if os.environ.get("HF_HOME"):
        add_candidate(Path(os.environ["HF_HOME"]))
    add_candidate(Path("/root/.cache/huggingface"))
    add_candidate(workspace / ".hf_home")

    for hf_home in candidates:
        if hf_hub_cache_complete(hf_home, model_id):
            os.environ["HF_HOME"] = str(hf_home)
            return str(hf_home)

    return os.environ.get("HF_HOME", str(workspace / ".hf_home"))


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
HIJACK_EN_PROMPT_NE_SHORT_PROBE_INFER = (
    HIJACK_REASONING_DIR / "hijack_reasoninig_en_prompt_ne_short_probe_infer.jsonl"
)
HIJACK_BYPASS_OUTPUT_PREFIX_INFER = (
    HIJACK_REASONING_DIR / "hijack_reasoninig_bypass_output_prefix_infer.jsonl"
)
HIJACK_PROMPT_ONLY_NO_THINK_INFER = (
    HIJACK_REASONING_DIR / "hijack_reasoninig_prompt_only_no_think_infer.jsonl"
)

HIJACK_DINKA_DIR = HIJACK_REASONING_DIR / "dinka"
HIJACK_BYPASS_DEEPSEEKV3_DIN = (
    HIJACK_DINKA_DIR / "hijack_reasoninig_bypass_deepseekv3_din.jsonl"
)
HIJACK_BYPASS_DEEPSEEKV3_DIN_INFER = (
    HIJACK_DINKA_DIR / "hijack_reasoninig_bypass_deepseekv3_din_infer.jsonl"
)
HIJACK_DINKA_PROBE_DINKA = (
    HIJACK_DINKA_DIR / "hijack_reasoning_bypass_deepseekv3-03_probe_dinka.jsonl"
)
HIJACK_DINKA_PROBE_DINKA_INFER = (
    HIJACK_DINKA_DIR / "hijack_reasoning_bypass_deepseekv3-03_probe_dinka_infer.jsonl"
)
HIJACK_DINKA_PROBE_DINKA_INFER_EN = (
    HIJACK_DINKA_DIR
    / "hijack_reasoning_bypass_deepseekv3-03_probe_dinka_infer_en.jsonl"
)


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

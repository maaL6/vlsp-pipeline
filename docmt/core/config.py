"""Run configuration (fields follow Lam/mau_ban_giao/run_config.template.json, all implemented)."""
# Split from lam_docmt/config.py (statements verbatim; tools/split_from_lam_docmt.py).
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields

@dataclass
class RunConfig:
    run_id: str = "TO_FILL"
    branch: str = "hy_a1_structure"
    owner: str = "Lam"
    backend: str = "hf"                  # "hf" | "mock"
    model: str = ""                      # HF model dir (hf backend)
    model_sha256: str = ""
    tokenizer_path: str = ""
    tokenizer_sha256: str = ""
    server_url: str = "http://127.0.0.1:8099"   # unused since llama.cpp was dropped; kept so old run_config.json files load
    # structure / budgets (draft section 2)
    source_cap: int = 768                # tokens of the unit payload, markers included
    evidence_cap: int = 512
    max_sequence: int = 4096
    margin: int = 32
    max_nodes: int = 8
    table_cap: int = 384
    max_table_nodes: int = 16
    signature_cap: int = 128
    recovery_cap: int = 256
    quality_recovery_rounds: int = 1
    recent_accepted_units: int = 2
    soft_terms: int = 8
    batch_size: int = 4
    # decoding
    do_sample: bool = False
    repetition_penalty: float = 1.05
    output_budget: str = "min(remaining_with_margin, max(256, ceil(2.5*source_tokens)+128))"
    prompt_style: str = "default"        # "default" | "delimiter" (see prompts.py)
    # features
    source_context: bool = False
    soft_glossary: bool = False
    target_history: bool = False
    selective_recovery: bool = True
    normalize_source: bool = True
    normalize_reference: bool = False
    reference_available_at_inference: bool = False
    glossary_path: str = ""
    seed: int = 42
    # hf backend
    hf_dtype: str = "float16"
    hf_device: str = "cuda"
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def sha256(self) -> str:
        d = self.to_dict()
        d.pop("run_id", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def apply(self, overrides: dict):
        names = {f.name: f for f in fields(self)}
        for k, v in overrides.items():
            if k not in names:
                raise KeyError(f"unknown config key {k}")
            cur = getattr(self, k)
            if isinstance(cur, bool) and isinstance(v, str):
                v = v.lower() in ("1", "true", "yes", "on")
            elif isinstance(cur, int) and not isinstance(cur, bool) and isinstance(v, str):
                v = int(v)
            elif isinstance(cur, float) and isinstance(v, str):
                v = float(v)
            setattr(self, k, v)
        return self

# Branch presets: change one component per ablation (contract section 2).
BRANCHES = {
    "hy_a1_structure": dict(source_context=False, soft_glossary=False, target_history=False),
    "hy_a1_source_context": dict(source_context=True, soft_glossary=False, target_history=False),
    "hy_a1_source_context_glossary": dict(source_context=True, soft_glossary=True, target_history=False),
    "hy_a1_source_context_history": dict(source_context=True, soft_glossary=False, target_history=True),
}

def make_config(branch: str, **overrides) -> RunConfig:
    cfg = RunConfig(branch=branch)
    cfg.apply(BRANCHES[branch])
    cfg.apply(overrides)
    return cfg

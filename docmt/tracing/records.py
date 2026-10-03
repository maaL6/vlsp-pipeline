"""Split from lam_docmt/trace.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

import json
import os
import platform
import sys

from ..core.hashing import utc_now
from .writer import TRACE_KEYS

def environment_text(backend_desc: dict | None = None, extra: dict | None = None) -> str:
    from importlib import metadata

    lines = [f"utc: {utc_now()}", f"python: {sys.version.split()[0]} ({sys.executable})",
             f"platform: {platform.platform()}", f"machine: {platform.machine()}",
             f"cpu_count: {os.cpu_count()}"]
    for pkg in ("tokenizers", "jinja2", "numpy", "gguf", "torch", "transformers", "accelerate",
                "safetensors", "sacrebleu", "pytest"):
        try:
            lines.append(f"package {pkg}: {metadata.version(pkg)}")
        except Exception:
            lines.append(f"package {pkg}: not installed")
    for var in ("CUDA_VISIBLE_DEVICES", "RAYON_NUM_THREADS", "OMP_NUM_THREADS"):
        if var in os.environ:
            lines.append(f"env {var}={os.environ[var]}")
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}: {v}")
    if backend_desc is not None:
        lines.append("backend: " + json.dumps(backend_desc, ensure_ascii=False, default=str))
    return "\n".join(lines) + "\n"

def make_trace_record(**kw) -> dict:
    """Ordered trace record; every TRACE_KEYS key present (missing ones = None), extras kept."""
    rec = {k: kw.pop(k, None) for k in TRACE_KEYS}
    rec.update(kw)
    return rec

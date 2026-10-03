"""Split from lam_docmt/trace.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

# -- hashes / time ---------------------------------------------------------------------------------
def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

_MODEL_FILE_EXT = (".safetensors", ".bin", ".gguf", ".json", ".model", ".txt", ".jinja", ".py")

def model_sha256(path) -> dict:
    """{"sha256", "files"}: a single file is hashed directly; a directory is hashed as the sorted
    list of "relpath<TAB>sha256" lines of its weight/config/tokenizer files."""
    p = Path(path)
    if p.is_file():
        s = sha256_file(p)
        return {"sha256": s, "files": {p.name: s}}
    if p.is_dir():
        files = {}
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix in _MODEL_FILE_EXT:
                files[str(f.relative_to(p))] = sha256_file(f)
        joined = "".join(f"{k}\t{v}\n" for k, v in sorted(files.items()))
        return {"sha256": sha256_text(joined), "files": files}
    return {"sha256": "", "files": {}}

def code_sha256(pkg_dir=None) -> dict:
    """sha256 of every docmt/**/*.py file plus a combined hash."""
    root = Path(pkg_dir) if pkg_dir else Path(__file__).resolve().parents[1]
    files = {}
    for f in sorted(root.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        files[str(f.relative_to(root.parent))] = sha256_file(f)
    joined = "".join(f"{k}\t{v}\n" for k, v in sorted(files.items()))
    return {"combined": sha256_text(joined), "files": files}

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

def read_jsonl(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn last line after a crash
    return out

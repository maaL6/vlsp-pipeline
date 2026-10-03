"""Split from lam_docmt/runtime.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DIRECTIONS = ("en-vi", "vi-en")

# -- manifest rows -----------------------------------------------------------------------------------
@dataclass
class DocItem:
    sample_id: str
    category: str
    direction: str
    src_lang: str
    tgt_lang: str
    source_path: Path
    rel_source: str = ""
    output_path: Optional[Path] = None
    rel_output: str = ""

def norm_directions(directions) -> list:
    if directions is None:
        return list(DIRECTIONS)
    if isinstance(directions, str):
        directions = [d.strip() for d in directions.split(",") if d.strip()]
    out = []
    for d in directions:
        if d not in DIRECTIONS:
            raise ValueError(f"unknown direction {d!r}; expected {DIRECTIONS}")
        if d not in out:
            out.append(d)
    return out

def expand_rows(rows, data_root, directions=None) -> list:
    """Doc-direction items. Rows with a "direction" (smoke6/pilot16) keep it and their
    source_path; rows without one (raw_dev_manifest) are expanded to every requested direction
    using <src>_path. Paths are relative to data_root. Only source paths are resolved."""
    dirs = norm_directions(directions)
    root = Path(data_root) if data_root is not None else Path(".")
    items = []
    for row in rows:
        sid = row["sample_id"]
        cat = row.get("category") or sid.split("/")[0]
        if row.get("direction"):
            todo = [row["direction"]] if row["direction"] in dirs else []
        else:
            todo = list(dirs)
        for d in todo:
            src, tgt = d.split("-")
            rel = row.get("source_path") if row.get("direction") == d and row.get("source_path") else row[f"{src}_path"]
            items.append(DocItem(sample_id=sid, category=cat, direction=d, src_lang=src, tgt_lang=tgt,
                                 source_path=root / rel, rel_source=str(rel)))
    return items

def prediction_relpath(direction: str, sample_id: str) -> str:
    return str(Path("predictions") / direction / Path(*sample_id.split("/")).with_suffix(".md"))

def read_source(path) -> str:
    """Exact raw text (no newline translation)."""
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()

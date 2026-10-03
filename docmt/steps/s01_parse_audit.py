"""Split from lam_docmt/parse.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

from .s01_parse import PARSER_VERSION, check_partition, parse_document, reconstruct

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def audit_document(src: str, nodes: list, tok=None) -> dict:
    """Per doc-direction audit record (used by parse_audit and the tests)."""
    errs = check_partition(src, nodes)
    rec = reconstruct(nodes)
    covered = sum(len(n.content_raw) - len(re.sub(r"\S", "", n.content_raw)) for n in nodes)
    total_nonws = len(src) - len(re.sub(r"\S", "", src))
    kinds = Counter(n.kind for n in nodes)
    scopes = []
    seen = {}
    for n in nodes:
        if n.scope_id not in seen:
            seen[n.scope_id] = {"scope_id": n.scope_id, "n_nodes": 0,
                                "first_line": n.content_raw.split("\n")[0].strip()[:100],
                                "first_node": n.node_id}
            scopes.append(seen[n.scope_id])
        seen[n.scope_id]["n_nodes"] += 1
    tables = {}
    for n in nodes:
        if n.table:
            g = n.table.get("group")
            t = tables.setdefault(g, {"group": g, "table_id": n.table.get("table_id"), "rows": set(), "cells": 0,
                                      "numeric_cells": 0, "copy_cells": 0, "translatable_cells": 0,
                                      "certain": n.table.get("certain"), "flags": set(), "payload_tokens": 0,
                                      "scope_id": n.scope_id, "first_node": n.node_id})
            t["rows"].add(n.table.get("row"))
            if n.kind == "table_cell":
                t["cells"] += 1
                t["numeric_cells"] += int(bool(n.table.get("numeric")))
                t["copy_cells"] += int(not n.translatable)
                t["translatable_cells"] += int(n.translatable)
                if tok is not None and n.translatable:
                    t["payload_tokens"] += tok.count(n.payload)
            t["flags"].update(n.flags)
    tables_out = []
    for t in tables.values():
        t["rows"] = len(t["rows"])
        t["flags"] = sorted(t["flags"])
        if tok is None:
            t["payload_tokens"] = None
        tables_out.append(t)
    flags = Counter(f for n in nodes for f in n.flags if not f.startswith("fallback_reason"))
    out = {
        "ok": not errs and rec == src and covered == total_nonws,
        "errors": errs,
        "roundtrip_exact": rec == src,
        "offsets_contiguous": not any(e.startswith(("gap_or_overlap", "offset", "end_offset")) for e in errs),
        "non_ws_chars": total_nonws,
        "non_ws_chars_in_content": covered,
        "n_chars": len(src),
        "source_sha256": _sha256(src),
        "n_nodes": len(nodes),
        "n_nodes_by_kind": dict(sorted(kinds.items())),
        "n_translatable": sum(1 for n in nodes if n.translatable),
        "n_copy": sum(1 for n in nodes if not n.translatable),
        "scopes": scopes,
        "tables": tables_out,
        "signature_names": [n.content_raw for n in nodes if n.kind == "signature_name"],
        "signature_lines": sum(1 for n in nodes if n.kind == "signature"),
        "unknown_count": kinds.get("unknown", 0),
        "low_confidence_count": sum(1 for n in nodes if n.confidence < 0.5),
        "flags": dict(sorted(flags.items())),
        "fallback_parse": any("fallback_parse" in n.flags for n in nodes),
        "heading_numbers": [n.protected.get("heading_number") for n in nodes
                            if n.kind == "article" and not n.scope_id.startswith("quote_")],
    }
    if tok is not None:
        toks = [tok.count(n.payload) for n in nodes if n.translatable]
        out["payload_tokens_total"] = sum(toks)
        out["payload_tokens_max_node"] = max(toks) if toks else 0
    return out

def _row_items(row: dict, both_directions: bool):
    """Expand a manifest row into (sample_id, direction, src_lang, source_relpath)."""
    sid = row.get("sample_id") or row.get("doc_id") or row.get("id")
    if "source_path" in row and not both_directions:
        src_lang = row.get("src_lang") or row.get("direction", "en-vi").split("-")[0]
        tgt = row.get("tgt_lang") or ("vi" if src_lang == "en" else "en")
        yield sid, row.get("direction") or f"{src_lang}-{tgt}", src_lang, row["source_path"]
        return
    for src_lang, tgt in (("en", "vi"), ("vi", "en")):
        key = f"{src_lang}_path"
        if key in row:
            yield sid, f"{src_lang}-{tgt}", src_lang, row[key]

def parse_audit(rows, data_root, tok, out_path, both_directions: bool = False) -> dict:
    """Parse every doc-direction of `rows` (SOURCE side only) and write a roundtrip report (JSON).

    rows: manifest dicts, either doc-direction rows ({sample_id, direction|src_lang, source_path}) or
    document rows ({sample_id, en_path, vi_path}) which are audited in both directions. Paths are
    relative to `data_root`. Returns the report dict."""
    data_root = Path(data_root)
    docs = []
    t0 = time.time()
    for row in rows:
        for sid, direction, src_lang, rel in _row_items(row, both_directions):
            path = data_root / rel
            src = path.read_bytes().decode("utf-8")
            t1 = time.time()
            nodes = parse_document(src, sid, src_lang, tok)
            rec = audit_document(src, nodes, tok)
            rec = {"sample_id": sid, "direction": direction, "src_lang": src_lang, "source_path": str(rel),
                   "parse_seconds": round(time.time() - t1, 3), **rec}
            docs.append(rec)
    kinds = Counter()
    flags = Counter()
    for d in docs:
        kinds.update(d["n_nodes_by_kind"])
        flags.update(d["flags"])
    totals = {
        "doc_directions": len(docs),
        "ok": sum(1 for d in docs if d["ok"]),
        "failed": [f'{d["sample_id"]}:{d["direction"]}' for d in docs if not d["ok"]],
        "roundtrip_exact": sum(1 for d in docs if d["roundtrip_exact"]),
        "offsets_contiguous": sum(1 for d in docs if d["offsets_contiguous"]),
        "fallback_parse": sum(1 for d in docs if d["fallback_parse"]),
        "nodes": sum(d["n_nodes"] for d in docs),
        "translatable_nodes": sum(d["n_translatable"] for d in docs),
        "copy_nodes": sum(d["n_copy"] for d in docs),
        "nodes_by_kind": dict(sorted(kinds.items())),
        "unknown_nodes": sum(d["unknown_count"] for d in docs),
        "low_confidence_nodes": sum(d["low_confidence_count"] for d in docs),
        "docs_with_tables": sum(1 for d in docs if d["tables"]),
        "table_groups": sum(len(d["tables"]) for d in docs),
        "docs_with_signature_name": sum(1 for d in docs if d["signature_names"]),
        "signature_names": sum(len(d["signature_names"]) for d in docs),
        "docs_with_attachment_or_appendix": sum(
            1 for d in docs if any(s["scope_id"].startswith(("attachment_", "appendix_")) for s in d["scopes"])),
        "docs_with_quote_scopes": sum(1 for d in docs if any(s["scope_id"].startswith("quote_") for s in d["scopes"])),
        "flags": dict(sorted(flags.items())),
    }
    report = {
        "status": "completed",
        "report": "parser_roundtrip",
        "parser_version": PARSER_VERSION,
        "utc_generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "timezone": "UTC; team UTC+7",
        "data_root": str(data_root),
        "source_side_only": True,
        "tokenizer_sha256": getattr(tok, "sha256", None),
        "code_sha256": _sha256(Path(__file__).read_text(encoding="utf-8")),
        "seconds": round(time.time() - t0, 2),
        "totals": totals,
        "documents": docs,
    }
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report

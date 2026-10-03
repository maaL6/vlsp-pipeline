"""Split from lam_docmt/validate.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

import collections
import re

from ..checks.common import ANY_MARKER, FFFD, _dedupe, _nfc, _ws
from ..checks.headings import compare_headings
from ..checks.loops import find_loops
from ..checks.numbers import compare_numbers
from .s07_validate_unit import _ratio_bounds

# ============================================================================ documents
def _length_info(src, out, src_lang, tgt_lang):
    sc, oc = _ws(src), _ws(out)
    lo, hi = _ratio_bounds(src_lang, tgt_lang)
    return {"src_chars": len(src), "out_chars": len(out),
            "ratio_raw": round(len(out) / max(len(src), 1), 3),
            "src_chars_collapsed": len(sc), "out_chars_collapsed": len(oc),
            "ratio": round(len(oc) / max(len(sc), 1), 3), "bounds": [lo, hi],
            "src_lines": src.count("\n") + 1, "out_lines": out.count("\n") + 1 if out else 0}

def audit_plain_output(src, out_text, src_lang, tgt_lang):
    """Audit of an output produced without units (legacy / B1 / reruns): heading numbers
    missing/extra, number multiset diff (values >= 4 digits listed with context), loops not in
    the source, U+FFFD, length ratio, internal marker leak. Heuristic source-vs-output checks:
    not a human review and not a metric."""
    src = src or ""
    out = out_text or ""
    hard, warnings = [], []
    if not out.strip():
        hard.append("empty_output")
    leak_out = ANY_MARKER.findall(out)
    leak_src = collections.Counter(ANY_MARKER.findall(src))
    leaks = [m for m in leak_out if leak_src[m] == 0]
    if leaks:
        hard.append("internal_marker_leak")
    heads = compare_headings(src, out)
    nums = compare_numbers(src, out, src_lang, tgt_lang, context=True)
    loops = find_loops(src, out)
    fffd = {"output": out.count(FFFD), "source": src.count(FFFD)}
    fffd["net"] = fffd["output"] - fffd["source"]
    length = _length_info(src, out, src_lang, tgt_lang)
    warnings += [f"heading_missing:{k}" for k in heads["missing"]]
    warnings += [f"heading_extra:{k}" for k in heads["extra"]]
    warnings += [f"heading_duplicate:{d['key']}" for d in heads["duplicate"]]
    if heads["not_as_heading"] or heads["fewer"]:
        warnings.append("heading_not_as_heading")
    if not heads["sequence_equal"]:
        warnings.append("heading_order")
    if nums["missing"] or nums["extra"]:
        warnings.append("number_diff")
    if nums["big_missing"] or nums["big_extra"]:
        warnings.append("number_diff_big")
    if loops["flagged"]:
        warnings.append("loop_not_in_source")
    if fffd["net"] > 0:
        warnings.append("fffd")
    if out.strip():
        if length["ratio"] < length["bounds"][0]:
            warnings.append("length_ratio_low")
        elif length["ratio"] > length["bounds"][1]:
            warnings.append("length_ratio_high")
    return {"kind": "heuristic_source_vs_output (not human review, not BLEU/chrF)",
            "src_lang": src_lang, "tgt_lang": tgt_lang, "hard": hard, "warnings": warnings,
            "headings": heads, "numbers": nums, "loops": loops, "fffd": fffd, "length": length,
            "markers": {"internal_marker_leak": len(leaks), "samples": leaks[:5]}}

def _piece_key(node_id):
    return [int(x) if x.isdigit() else x for x in re.split(r"[.]", node_id)]

def check_document(src, out_text, nodes, selected, unresolved, ns, src_lang, tgt_lang) -> dict:
    """Document-level check after assembly. HARD: coverage_missing|coverage_empty:<node_id>
    (translatable node without a non-empty selected translation), internal_marker_leak (our
    namespace `ns`, incl. broken fragments), source_marker_lost:<marker>, copy_node_missing:<id>
    (copied raw text absent from the assembled output). Plus the plain audit (headings sequence,
    numbers, loops, U+FFFD, length) as warnings and counts."""
    out = out_text or ""
    selected = selected or {}
    unresolved = list(unresolved or [])
    hard, warnings = [], []
    trans = [n for n in nodes if n.translatable]
    missing, empty = [], []
    for n in trans:
        t = selected.get(n.node_id)
        if t is None:
            pieces = sorted((k for k in selected if k.startswith(n.node_id + ".")), key=_piece_key)
            if pieces:
                t = " ".join(selected[k] or "" for k in pieces)
        if t is None:
            missing.append(n.node_id)
        elif not str(t).strip():
            empty.append(n.node_id)
    hard += [f"coverage_missing:{i}" for i in missing] + [f"coverage_empty:{i}" for i in empty]
    # markers
    own_full = re.compile(r"\[" + re.escape(ns) + r"_\d{6}\]")
    own_frag = re.compile(r"\[?" + re.escape(ns) + r"_\d{1,6}\]?")
    leak_full = max(0, len(own_full.findall(out)) - len(own_full.findall(src)))
    leak_frag = max(0, len(own_frag.findall(out)) - len(own_frag.findall(src)))
    if leak_full or leak_frag:
        hard.append("internal_marker_leak")
    src_markers = collections.Counter(ANY_MARKER.findall(src))
    lost = [m for m, c in sorted(src_markers.items()) if out.count(m) < c and not own_full.fullmatch(m)]
    hard += [f"source_marker_lost:{m}" for m in lost]
    # copied (non-translatable) nodes must appear verbatim, in order
    pos, copy_missing, out_nfc = 0, [], _nfc(out)
    for n in nodes:
        if n.translatable or not (n.content_raw or "").strip():
            continue
        i = out_nfc.find(_nfc(n.content_raw), pos)
        if i < 0:
            copy_missing.append(n.node_id)
        else:
            pos = i + len(_nfc(n.content_raw))
    hard += [f"copy_node_missing:{i}" for i in copy_missing]
    audit = audit_plain_output(src, out, src_lang, tgt_lang)
    hard += [h for h in audit["hard"] if h not in hard and h != "internal_marker_leak"]
    warnings += audit["warnings"]
    if unresolved:
        warnings.append("unresolved_units")
    status = "failed" if not out.strip() else (
        "completed_with_unresolved" if (unresolved or hard) else "completed")
    kinds = collections.Counter(n.kind for n in nodes)
    return {"hard": _dedupe(hard), "warnings": _dedupe(warnings), "status_hint": status,
            "coverage": {"translatable": len(trans), "missing": missing, "empty": empty,
                         "covered": len(trans) - len(missing) - len(empty),
                         "ratio": round((len(trans) - len(missing) - len(empty)) / max(len(trans), 1), 4)},
            "markers": {"namespace": ns, "internal_leak_full": leak_full, "internal_leak_fragments": leak_frag,
                        "source_markers": dict(src_markers), "source_markers_lost": lost},
            "copy_nodes_missing": copy_missing,
            "headings": audit["headings"], "numbers": audit["numbers"], "loops": audit["loops"],
            "fffd": audit["fffd"], "length": audit["length"],
            "counts": {"nodes": len(nodes), "translatable": len(trans), "copy": len(nodes) - len(trans),
                       "by_kind": dict(kinds), "selected": len(selected), "unresolved": len(unresolved)},
            "unresolved": unresolved}


# ---- document status (moved from engine/doc_job.py _DocJob.finish) ---------------------------------
def decide_status(op_error: bool, out_text: str, unresolved_units, empty_work, uncovered, doc_hard):
    """(status, error): failed on an operational error or an empty document; completed_with_unresolved
    when any unit/node is unresolved, uncovered, or a document-level hard failure remains."""
    if op_error:
        return "failed", "operational_error"
    if not out_text.strip():
        return "failed", "empty_output_document"
    if unresolved_units or empty_work or uncovered or doc_hard:
        return "completed_with_unresolved", None
    return "completed", None

"""Validator: deterministic HARD failures + heuristic WARNINGS, source vs output only.

- HARD failures are deterministic and trigger the one quality-recovery round.
- WARNINGS are heuristics (status `needs_review`); they are never proof of a meaning error
  (a regex is never used as firm evidence of a wrong meaning).
- Nothing here reads reference/gold text: every check compares the SOURCE with the OUTPUT.

Public API: check_unit, check_document, audit_plain_output, number_tokens.
Helpers used by tools/audit_existing.py: compare_numbers, compare_headings, find_loops,
heading_list, display_heading, display_value.
"""
from __future__ import annotations

import re

from ..checks.common import ANY_MARKER, FFFD, _dedupe, _nfc
from ..checks.headings import _HEADING_TYPES, _has_heading, _hvalue
from ..checks.lexical import _untranslated, modal_count, negation_count
from ..checks.loops import find_loops
from ..checks.numbers import compare_numbers
from ..core.datatypes import UnitCheck

RATIO_BOUNDS = {("en", "vi"): (0.45, 2.6), ("vi", "en"): (0.35, 2.0)}  # chars out/src

MIN_RATIO_CHARS = 20          # skip length-ratio warning for shorter source payloads

# ============================================================================ check_unit
def _ratio_bounds(src_lang, tgt_lang, cfg=None):
    extra = getattr(cfg, "extra", None) or {}
    custom = (extra.get("validator") or {}).get("ratio_bounds") if isinstance(extra, dict) else None
    if custom and f"{src_lang}-{tgt_lang}" in custom:
        lo, hi = custom[f"{src_lang}-{tgt_lang}"]
        return float(lo), float(hi)
    return RATIO_BOUNDS.get((src_lang, tgt_lang), (0.35, 2.6))

_LABEL_CORE = r"\d{1,3}(?:\.\d{1,2})*|(?:dd|[a-zđ])\d{0,3}|[ivxlc]{1,6}"
_SEG_LABEL = re.compile(r"^[“\"'‘\[]*\s*(?:\(\s*("+_LABEL_CORE+r")\s*\)|("+_LABEL_CORE+r")\s*[.)\-–:])", re.I)

def _label_of(text):
    m = _SEG_LABEL.match(_nfc(text).strip())
    if not m:
        return None
    return (m.group(1) or m.group(2)).lower().replace("dd", "đ")

def _inline_label_pattern(core):
    # Vietnamese đ is conventionally rendered dd in EN labels; d is distinct.
    value=re.escape(core).replace('đ',r'(?:đ|dd)')
    return re.compile(r'(?<!\w)\(?\s*'+value+r'\s*[.)\-–:]',re.I)

def source_sub_labels(payload):
    # Only explicit sub-point enumerators after a boundary, not prose references
    # such as "points a, d2, g1". The parser may keep these inside a parent node.
    return list(dict.fromkeys(m.group(1) for m in re.finditer(
        r'(?:^|[;:\n])\s*((?:dd|[a-zđ])\d{1,3}\))',payload,re.I)))

def check_unit(unit, nodes_by_id, result, parsed, src_lang, tgt_lang, cfg=None) -> UnitCheck:
    """Checks one generated unit. HARD: ParseResult.hard, no_eos, protected_heading_lost:<id>,
    table_cell_multiline:<id>. WARNINGS (needs_review): length_ratio_low|high:<id>,
    number_mismatch:<id>, label_changed:<id>, possible_untranslated:<id>, loop_not_in_source,
    fffd, negation_count_diff, modal_count_diff."""
    hard = list(parsed.hard) if parsed is not None else ["empty_output"]
    warnings = []
    details = {"stop_type": getattr(result, "stop_type", None), "eos": bool(getattr(result, "eos", False)),
               "nodes": {}}
    if not getattr(result, "eos", False):
        hard.append("no_eos")
    if getattr(result, "error", None):
        details["operational_error"] = result.error
    lo, hi = _ratio_bounds(src_lang, tgt_lang, cfg)
    segs = parsed.segments if parsed is not None else {}
    src_parts, out_parts = [], []
    for nid in unit.node_ids:
        node = nodes_by_id.get(nid)
        seg = segs.get(nid)
        if node is None or seg is None or not seg.strip():
            details["nodes"][nid] = {"segment": "absent_or_empty"}   # covered by parse hard failures
            continue
        payload = node.payload or ""
        nd = {}
        src_parts.append(payload)
        out_parts.append(seg)
        # HARD: protected heading number lost
        prot = node.protected or {}
        for label in source_sub_labels(payload):
            core=_label_of(label)
            if core and not _inline_label_pattern(core).search(seg):
                hard.append(f'protected_inline_label_lost:{nid}:{core}')
        # Verify every legal heading/label retained inside a block, independently of
        # target newlines. No line-i to source-node-i alignment is fabricated.
        for member in prot.get('block_members',[]):
            mp=member.get('protected',{});mid=member['node_id']
            mt,mn=mp.get('heading_type'),mp.get('heading_number')
            if mt in _HEADING_TYPES and mn not in (None,''):
                value=_hvalue(str(mn))
                if value and _has_heading(payload,mt,value) and not _has_heading(seg,mt,value):
                    hard.append(f'protected_heading_lost:{mid}')
            label=mp.get('label')
            core=_label_of(str(label)) if label else None
            # Bullets and non-numbered labels are not numbered clauses. Two
            # failed parses (None == None) must not enter the numeric regex.
            if core and _label_of(member.get('payload',''))==core:
                # Labels may move inline, but the value and delimiter must survive.
                pat=_inline_label_pattern(core)
                if pat.search(payload) and not pat.search(seg):hard.append(f'protected_block_label_lost:{mid}')
        htype, hnum = prot.get("heading_type"), prot.get("heading_number")
        if htype in _HEADING_TYPES and hnum not in (None, ""):
            want = _hvalue(str(hnum))
            if want and _has_heading(payload, htype, want):
                if not _has_heading(seg, htype, want):
                    hard.append(f"protected_heading_lost:{nid}")
                    nd["heading_lost"] = f"{htype}:{want}"
            elif want:
                nd["heading_check"] = "skipped: heading number not in this payload"
        # HARD: a single-line table cell translated into several lines
        if (node.kind == "table_cell" or node.table is not None) and "\n" not in node.content_raw.strip() \
                and "\n" in seg.strip():
            hard.append(f"table_cell_multiline:{nid}")
        # warnings
        if len(payload) >= MIN_RATIO_CHARS:
            r = len(seg.strip()) / len(payload)
            nd["ratio"] = round(r, 3)
            if r < lo:
                warnings.append(f"length_ratio_low:{nid}")
            elif r > hi:
                warnings.append(f"length_ratio_high:{nid}")
        num = compare_numbers(payload, seg, src_lang, tgt_lang)
        if num["missing"] or num["extra"]:
            warnings.append(f"number_mismatch:{nid}")
            nd["numbers"] = {"missing": [d["value"] for d in num["missing"]],
                             "extra": [d["value"] for d in num["extra"]]}
        want_label = prot.get("label")
        if want_label:
            core = _label_of(str(want_label))
            if core and _label_of(payload) == core:
                got = _label_of(seg)
                if got != core:
                    warnings.append(f"label_changed:{nid}")
                    nd["label"] = {"src": want_label, "out": got}
        why = _untranslated(payload, seg, tgt_lang)
        if why:
            warnings.append(f"possible_untranslated:{nid}")
            nd["untranslated"] = why
        if nd:
            details["nodes"][nid] = nd
    # unit-level warnings
    out_text = getattr(result, "text", "") or ""
    if out_text.count(FFFD) > (unit.payload or "").count(FFFD):
        warnings.append("fffd")
        details["fffd"] = out_text.count(FFFD)
    loops = find_loops(ANY_MARKER.sub(" ", unit.payload or ""), ANY_MARKER.sub(" ", out_text))
    if loops["flagged"]:
        # Severe consecutive substring degeneration only. Repeated headings/line boilerplate
        # remain warnings; do not deduplicate legitimate translated legal clauses.
        table_unit=any(nodes_by_id[nid].table is not None or nodes_by_id[nid].kind=="table_cell"
                       for nid in unit.node_ids if nid in nodes_by_id)
        minimum_chars=160 if table_unit else 40
        severe=[x for x in loops["substring"] if x["flagged"] and x["class"]=="text"
                and x["unit_len"]>=8 and x["chars"]>=minimum_chars and x["reps"]>=5]
        if severe and (getattr(cfg,"extra",{}) or {}).get("degeneration_retry"):
            hard.append("degenerate_repetition")
        warnings.append("loop_not_in_source")
        details["loops"] = [x for x in loops["substring"] + loops["lines"] if x["flagged"]][:5]
    if src_parts:
        s, o = "\n".join(src_parts), "\n".join(out_parts)
        ns, no = negation_count(s, src_lang), negation_count(o, tgt_lang)
        if (ns == 0) != (no == 0) or abs(ns - no) >= 2:
            warnings.append("negation_count_diff")
        ms, mo = modal_count(s, src_lang, "strict"), modal_count(o, tgt_lang, "lenient")
        if (ms >= 1 and mo == 0) or ms - mo >= 2:
            warnings.append("modal_count_diff")
        details["negation"] = {"src": ns, "out": no}
        details["modal"] = {"src_strict": ms, "out_lenient": mo}
    return UnitCheck(hard=_dedupe(hard), warnings=_dedupe(warnings), details=details)


# ---- attempt verdict (moved from engine/doc_job.py _DocJob._evaluate) ------------------------------
def combine_hard(parse_hard, validator_hard, eos: bool, leaks) -> list:
    """Hard failures of one attempt: parse_output + check_unit + no_eos + internal marker leaks."""
    hard = list(parse_hard) + list(validator_hard)
    if not eos:
        hard.append("no_eos")
    hard += [f"internal_marker_leak:{n}" for n in leaks]
    return _dedupe(hard)

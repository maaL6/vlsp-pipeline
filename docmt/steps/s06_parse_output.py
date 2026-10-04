"""Split from lam_docmt/units.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

import re

from ..core.datatypes import ParseResult
from .s03_pack import MARKER_RE, _ns_of

# ------------------------------------------------------------------------------- output parsing
def _near_miss_re(ns: str):
    # own-namespace marker remnants: "[SEG_12]", "[ SEG _000001", "seg_000001]" (any case)
    return re.compile(r"\[ *" + ns + r" *_[ \d]{0,8}\]?|(?<![\w\[])" + ns + r"_\d{3,}\]?",
                      re.IGNORECASE)

def parse_output(unit: Unit, text: str) -> ParseResult:
    """Split a model output into node segments using the unit's markers.

    Single node: the whole output. Several nodes and unit.markers == []: "newline" mode, see
    _parse_lines. Several nodes with markers: the rules below.

    Hard failures: empty_output, marker_missing:<id>, marker_duplicate:<id>, marker_order,
    marker_unexpected:<m>, unassigned_text, segment_empty:<id>. Segments are stripped; only
    non-empty segments are returned. Markers of another namespace stay in the text. For
    duplicates the first non-empty occurrence is kept. Own-namespace remnants that are not exact
    markers (e.g. '[SEG_12]') and are not in the source are removed and reported as unexpected."""
    text = text or ""
    nids = list(unit.node_ids)
    if not nids:
        raise ValueError(f"parse_output: unit {unit.unit_id} has no nodes")
    if len(nids) == 1:
        seg = text.strip()
        if not seg:
            return ParseResult({}, ["empty_output"], {"mode": "single"})
        return ParseResult({nids[0]: seg}, [], {"mode": "single"})
    if not unit.markers:                    # multi-node unit without markers = "newline" mode
        return _parse_lines(nids, text)
    if len(unit.markers) != len(nids):
        raise ValueError(f"parse_output: unit {unit.unit_id} has {len(nids)} nodes, "
                         f"{len(unit.markers)} markers")
    ns = _ns_of(unit.markers[0])
    pos_of = {m: i for i, m in enumerate(unit.markers)}
    extra = {"mode": "markers", "ns": ns, "found": 0, "unexpected": [], "dropped": [],
             "duplicates": {}, "near_miss": []}
    if not text.strip():
        return ParseResult({}, ["empty_output"], extra)

    own = [m for m in MARKER_RE.finditer(text) if m.group(1) == ns]
    extra["found"] = len(own)
    hard = []
    prefix = text[:own[0].start()] if own else text
    if prefix.strip():
        hard.append("unassigned_text")

    occ = {}                                                    # unit position -> [segment texts]
    seq = []
    unexpected = []
    for k, m in enumerate(own):
        end = own[k + 1].start() if k + 1 < len(own) else len(text)
        seg = text[m.end():end]
        p = pos_of.get(m.group(0))
        if p is None:
            unexpected.append(m.group(0))
            if seg.strip():
                extra["dropped"].append(seg.strip())
            continue
        occ.setdefault(p, []).append(seg)
        seq.append(p)

    source_wo_markers = re.sub(re.escape(f"[{ns}_") + r"\d{6}\]", " ", unit.payload)
    near = _near_miss_re(ns)

    def clean(s):
        def repl(mm):
            if mm.group(0) in source_wo_markers:
                return mm.group(0)
            extra["near_miss"].append(mm.group(0))
            return ""
        return near.sub(repl, s).strip()

    segments = {}
    missing, dup, empty = [], [], []
    for p, nid in enumerate(nids):
        segs = occ.get(p)
        if not segs:
            missing.append(nid)
            continue
        cleaned = [clean(s) for s in segs]
        if len(segs) > 1:
            dup.append(nid)
            extra["duplicates"][nid] = [s for s in cleaned[1:] if s]
        chosen = next((s for s in cleaned if s), "")
        if chosen:
            segments[nid] = chosen
        else:
            empty.append(nid)
    hard += [f"marker_missing:{nid}" for nid in missing]
    hard += [f"marker_duplicate:{nid}" for nid in dup]
    if any(b < a for a, b in zip(seq, seq[1:])):
        hard.append("marker_order")
    hard += [f"segment_empty:{nid}" for nid in empty]
    for m in dict.fromkeys(unexpected + extra["near_miss"]):
        hard.append(f"marker_unexpected:{m}")
    extra["unexpected"] = list(dict.fromkeys(unexpected))
    return ParseResult(segments, hard, extra)

def _parse_lines(nids, text: str) -> ParseResult:
    """Output of a multi-node unit sent without markers (payload_delimiter "newline", s03_pack).

    The payload had one node per line, so the output must have exactly one non-empty line per
    node, in order: line i -> nids[i] (stripped). Blank lines are ignored (the model sometimes
    adds an empty line between paragraphs). Any other count of non-empty lines -- the model merged
    two nodes into one line, split one node over two lines, or dropped/added a line -- cannot be
    mapped safely, so nothing is assigned and the hard failure is
    line_count_mismatch:<got>/<want>; s08 then re-translates each node of the unit alone (as it
    does for any failed newline unit). A matching count does not prove alignment: a split plus
    a merge keeps the count; the heading check in s07 catches the shifts that move a heading. Measured on the public test: 21% of first-pass units (A: 12-35% marker
    failures). Content checks (numbers, headings, loops ...) run afterwards in s07 as usual."""
    extra = {"mode": "lines", "found": 0}
    if not text.strip():
        return ParseResult({}, ["empty_output"], extra)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    extra["found"] = len(lines)
    if len(lines) != len(nids):
        return ParseResult({}, [f"line_count_mismatch:{len(lines)}/{len(nids)}"], extra)
    return ParseResult(dict(zip(nids, lines)), [], extra)

def merge_child_segments(parent: Unit, children, child_segments: dict) -> dict:
    """Map recovery-children segments back to the parent's node ids. A parent node gets its own
    segment, or the ' '-joined segments of all its pieces ('<id>.1', '<id>.1.2' ...) in order;
    a node with any missing/empty piece is left out (caller reports it)."""
    order = [nid for ch in children for nid in ch.node_ids]
    out = {}
    for nid in parent.node_ids:
        parts = [x for x in order if x == nid or x.startswith(nid + ".")]
        texts = [child_segments.get(x, "") for x in parts]
        if parts and all(t.strip() for t in texts):
            out[nid] = " ".join(t.strip() for t in texts)
    return out


# ---- internal marker leaks (moved from engine/doc_job.py _DocJob._evaluate) ------------------------
def leak_pattern(ns: str):
    """Regex of this document's internal markers ([SEG_000123] or [SEGxxxx_000123])."""
    return re.compile(r"\[" + re.escape(ns) + r"_\d{1,8}\]")


def strip_internal_markers(segments: dict, leak_re):
    """(segments without internal markers, node ids that leaked one). A leak is a hard failure."""
    segs, leaks = {}, []
    for nid, txt in (segments or {}).items():
        txt = txt or ""
        if leak_re.search(txt):
            txt = re.sub(r"[ \t]{2,}", " ", leak_re.sub("", txt)).strip()
            leaks.append(nid)
        segs[nid] = txt
    return segs, leaks

"""Split from lam_docmt/units.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

import hashlib
import re

from ..core.datatypes import Unit
from .s03_split import _FOLD, _check_partition, _limits, _run_key

MARKER_RE = re.compile(r"\[(SEG[0-9A-F]{0,4})_(\d{6})\]")

_NS_FULL = re.compile(r"SEG[0-9A-F]{0,4}")

MAX_MARKER_INDEX = 999_999

def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# ------------------------------------------------------------------------------------ markers
def marker_namespace(src: str) -> str:
    """'SEG', or 'SEG' + 4 upper-case hex of sha256(src) when '[SEG_' already occurs in the source.

    The source is also checked after folding the characters normalize() rewrites (payloads are
    normalized). If the hashed namespace itself occurs (practically impossible) the next 4-hex
    window of the same digest is used."""
    folded = src.translate(_FOLD)

    def taken(ns):
        probe = f"[{ns}_"
        return probe in src or probe in folded

    if not taken("SEG"):
        return "SEG"
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest().upper()
    for i in range(0, len(digest) - 3, 4):
        ns = "SEG" + digest[i:i + 4]
        if not taken(ns):
            return ns
    raise ValueError("marker_namespace: no free namespace for this source")

def marker(ns: str, index: int) -> str:
    """f"[{ns}_{index:06d}]"; index = position of the node in the document's node list."""
    if not _NS_FULL.fullmatch(ns or ""):
        raise ValueError(f"bad marker namespace {ns!r}")
    if not isinstance(index, int) or not 0 <= index <= MAX_MARKER_INDEX:
        raise ValueError(f"marker index out of range: {index!r}")
    return f"[{ns}_{index:06d}]"

def _ns_of(marker_str: str) -> str:
    m = MARKER_RE.fullmatch(marker_str)
    if not m:
        raise ValueError(f"not a marker: {marker_str!r}")
    return m.group(1)

def _index_of(marker_str: str) -> int:
    m = MARKER_RE.fullmatch(marker_str)
    if not m:
        raise ValueError(f"not a marker: {marker_str!r}")
    return int(m.group(2))

def render_payload(unit_nodes, markers) -> str:
    """'\\n'.join(f"{m} {n.payload}"); a single node is sent as its bare payload (no marker)."""
    nodes = list(unit_nodes)
    if not nodes:
        raise ValueError("render_payload: empty unit")
    if len(nodes) == 1:
        return nodes[0].payload
    markers = list(markers or [])
    if len(markers) != len(nodes):
        raise ValueError(f"render_payload: {len(nodes)} nodes but {len(markers)} markers")
    return "\n".join(f"{m} {n.payload}" for m, n in zip(markers, nodes))

def _greedy(items, count, cap: int, max_n: int):
    """Greedy packing in order. Yields (group, oversize); an item that alone exceeds cap is alone."""
    cur = []
    for it in items:
        if count([it]) > cap:
            if cur:
                yield cur, False
                cur = []
            yield [it], True
            continue
        cand = cur + [it]
        if len(cand) <= max_n and count(cand) <= cap:
            cur = cand
        else:
            if cur:
                yield cur, False
            cur = [it]
    if cur:
        yield cur, False

def pack_units(nodes, doc_id, tok, cfg, ns) -> list:
    """Pack translatable nodes (document order) into units.

    Text units: <= cfg.max_nodes nodes and <= cfg.source_cap tokens (payload incl. markers).
    Table units: one table group, <= cfg.table_cap tokens, <= cfg.max_table_nodes cells.
    Signature units: one signature block, <= cfg.signature_cap tokens, <= cfg.max_nodes nodes.
    Units never cross a scope / table group / signature block boundary; copy nodes are not sent
    (a copy table cell or signature line still ends a text run). A node over its cap is packed
    alone as kind "single". Marker index = position of the node in `nodes`."""
    nodes = list(nodes)
    _check_partition(nodes)
    runs = []                                                   # [(kind, [positions])]
    run_key = None
    for i, n in enumerate(nodes):
        key = _run_key(n)
        if not n.translatable:
            neutral = key[0] == "text" and run_key is not None and key[1] == run_key[1]
            if run_key is not None and key != run_key and not neutral:
                run_key = None
            continue
        if key != run_key:
            runs.append((key[0], []))
            run_key = key
        runs[-1][1].append(i)

    def count(group):
        g_nodes = [nodes[p] for p in group]
        marks = [marker(ns, p) for p in group] if len(group) > 1 else []
        return tok.count(render_payload(g_nodes, marks))

    units = []
    for kind, positions in runs:
        cap, max_n = _limits(kind, cfg)
        for group, oversize in _greedy(positions, count, cap, max_n):
            g_nodes = [nodes[p] for p in group]
            marks = [marker(ns, p) for p in group] if len(group) > 1 else []
            payload = render_payload(g_nodes, marks)
            span = "".join(nodes[p].raw for p in range(group[0], group[-1] + 1))
            units.append(Unit(
                unit_id=f"{doc_id}:u{len(units):04d}", doc_id=doc_id, scope_id=g_nodes[0].scope_id,
                kind="single" if oversize else kind, node_ids=[n.node_id for n in g_nodes],
                markers=marks, payload=payload, source_tokens=tok.count(payload),
                raw_start=g_nodes[0].raw_start, raw_end=g_nodes[-1].raw_end, raw_sha256=_sha(span)))
    return units

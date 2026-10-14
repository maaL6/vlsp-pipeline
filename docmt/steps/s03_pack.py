from __future__ import annotations

import hashlib
import re

from ..core.datatypes import Unit
from .s03_split import _FOLD, _check_partition, _limits, _run_key, ends_at_period

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

# ------------------------------------------------------------------------- payload delimiter
# How the nodes of a multi-node unit are separated in the prompt (cfg.extra["payload_delimiter"]):
#
#   "marker"  (default when the key is missing)
#       [SEG_000010] Pursuant to ...
#       [SEG_000011] At the request of ...
#     s06 splits the output on the markers; a lost/extra marker is a hard failure.
#
#   "newline" (configs/hy_mt_vlsp26.json)
#       Pursuant to ...
#       At the request of ...
#     Nodes are joined by '\n' only and s06 maps output line i -> node i; a different number of
#     non-empty lines is the hard failure line_count_mismatch. In s08 every failed newline unit
#     (line count or any other hard failure) is re-sent node by node, since a retry joined by
#     newlines again can be mis-aligned. Node payloads never contain '\n' (s02 unwraps them),
#     so one node = one line.
#
# Internally a unit sent with "newline" has Unit.markers == [] even when it holds several nodes;
# that is how render_payload (here) and parse_output (s06) recognise the newline format, so no
# other step needs the config.

PAYLOAD_DELIMITERS = ("marker", "newline")

def payload_delimiter(cfg) -> str:
    """cfg.extra["payload_delimiter"]: "marker" (default; missing key or no extra) or "newline"."""
    value = (getattr(cfg, "extra", None) or {}).get("payload_delimiter", "marker")
    if value not in PAYLOAD_DELIMITERS:
        raise ValueError(f"extra.payload_delimiter must be one of {PAYLOAD_DELIMITERS}, got {value!r}")
    return value

def use_markers(cfg) -> bool:
    return payload_delimiter(cfg) == "marker"

def unit_markers(ns: str, indices, cfg) -> list:
    """Markers of a unit whose nodes have these marker indices: one per node for a multi-node unit
    in "marker" mode; [] for a single node (always sent bare) and for any unit in "newline" mode."""
    indices = list(indices)
    if len(indices) > 1 and use_markers(cfg):
        return [marker(ns, i) for i in indices]
    return []

def render_payload(unit_nodes, markers) -> str:
    """Text sent to the model for one unit.

    - one node: its bare payload (never a marker);
    - several nodes + markers: '\\n'.join(f"{m} {n.payload}") ("marker" mode);
    - several nodes, markers == []: '\\n'.join(n.payload) ("newline" mode)."""
    nodes = list(unit_nodes)
    if not nodes:
        raise ValueError("render_payload: empty unit")
    if len(nodes) == 1:
        return nodes[0].payload
    markers = list(markers or [])
    if not markers:
        return "\n".join(n.payload for n in nodes)
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

def _period_greedy(items, nodes, count, cap, max_n):
    """Close text chunks only at sentence periods; node/token limits are soft.

    The final chunk of an actual scope/table/signature boundary is retained whole even if
    the source ends without a period. Never cross such a source-structure boundary.
    """
    current=[]
    for it in items:
        candidate=current+[it]
        if current and (count(candidate)>cap or len(candidate)>max_n) and ends_at_period(nodes[current[-1]].payload):
            yield current,count(current)>cap
            current=[]
        current.append(it)
    if current:yield current,count(current)>cap

def pack_units(nodes, doc_id, tok, cfg, ns) -> list:
    """Pack translatable nodes (document order) into units.

    Text units: <= cfg.max_nodes nodes and <= cfg.source_cap tokens (payload incl. markers).
    Table units: one table group, <= cfg.table_cap tokens, <= cfg.max_table_nodes cells.
    Signature units: one signature block, <= cfg.signature_cap tokens, <= cfg.max_nodes nodes.
    Units never cross a scope / table group / signature block boundary; copy nodes are not sent
    (a copy table cell or signature line still ends a text run). A node over its cap is packed
    alone as kind "single". Marker index = position of the node in `nodes`. Token caps count the
    rendered payload, so in "newline" mode (no marker tokens) a unit can hold slightly more text."""
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
        marks = unit_markers(ns, group, cfg)
        return tok.count(render_payload(g_nodes, marks))

    units = []
    for kind, positions in runs:
        cap, max_n = _limits(kind, cfg)
        # A prose block is already an exact source span. Accept its whole translation;
        # do not ask the model to preserve one line per original parser node.
        groups = (([p],count([p])>cap) for p in positions) if kind=='text' and (cfg.extra or {}).get('legal_prose_blocks') else (_period_greedy(positions,nodes,count,cap,max_n)
                  if kind == "text" and (cfg.extra or {}).get("strict_period_packing")
                  else _greedy(positions,count,cap,max_n))
        for group, oversize in groups:
            g_nodes = [nodes[p] for p in group]
            marks = unit_markers(ns, group, cfg)
            payload = render_payload(g_nodes, marks)
            span = "".join(nodes[p].raw for p in range(group[0], group[-1] + 1))
            units.append(Unit(
                unit_id=f"{doc_id}:u{len(units):04d}", doc_id=doc_id, scope_id=g_nodes[0].scope_id,
                kind="single" if oversize else kind, node_ids=[n.node_id for n in g_nodes],
                markers=marks, payload=payload, source_tokens=tok.count(payload),
                raw_start=g_nodes[0].raw_start, raw_end=g_nodes[-1].raw_end, raw_sha256=_sha(span)))
    return units

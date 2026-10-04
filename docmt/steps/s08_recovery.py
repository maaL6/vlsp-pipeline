"""Split from lam_docmt/runtime.py, lam_docmt/units.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

from ..core.codes import _code
from ..core.datatypes import Unit
from .s03_pack import _greedy, _sha, render_payload, unit_markers
from .s03_split import split_node

# Hard failures meaning "the output could not be mapped back to the unit's nodes". They send a
# multi-node unit to independent recovery (each node alone). line_count_mismatch is the
# "newline"-mode counterpart of the marker failures (see s03_pack, payload delimiter).
MARKER_CODES = ("marker_missing", "marker_duplicate", "marker_order", "marker_unexpected",
                "unassigned_text", "segment_empty", "internal_marker_leak", "line_count_mismatch")

def is_marker_failure(hard) -> bool:
    return any(_code(h) in MARKER_CODES for h in hard)

# ------------------------------------------------------------------------------------- recovery
def _span_text(nodes_by_id, a: int, b: int):
    """Raw text of [a, b) rebuilt from nodes_by_id (None if a gap is not covered)."""
    by_start = {}
    for n in nodes_by_id.values():
        if a <= n.raw_start < b and n.raw_end <= b:
            cur = by_start.get(n.raw_start)
            if cur is None or n.raw_end > cur.raw_end:
                by_start[n.raw_start] = n
    out, p = [], a
    while p < b:
        n = by_start.get(p)
        if n is None or n.raw_end <= p:
            return None
        out.append(n.raw)
        p = n.raw_end
    return "".join(out)

def _halves(items, count):
    best = None
    for k in range(1, len(items)):
        worst = max(count(items[:k]), count(items[k:]))
        if best is None or worst < best[0]:
            best = (worst, k)
    k = best[1]
    return [items[:k], items[k:]]

def recovery_units(unit, nodes_by_id, tok, cfg, ns, independent: bool) -> list:
    """Units for the single quality-recovery round of `unit` (re-translation from source).

    independent=True (marker / line-count failure): single-node pieces <= recovery_cap, kind
    "single", no markers.
    independent=False: nodes > cfg.recovery_cap tokens are split first (pieces added to
    nodes_by_id), then regrouped in order into units <= cfg.recovery_cap tokens (markers included;
    in "newline" mode the regrouped units are newline-joined like the first pass),
    kind "recovery"; if that would rebuild the same multi-node unit, it is cut in two halves so the
    retry is not an identical request. Ids '<unit_id>.r1', '.r2' ...; split_from = unit.unit_id.
    Marker index = position of the node in nodes_by_id (insertion order)."""
    try:
        parents = [nodes_by_id[nid] for nid in unit.node_ids]
    except KeyError as e:
        raise ValueError(f"recovery_units: node {e} of {unit.unit_id} not in nodes_by_id") from None
    work = []
    for n in parents:
        pieces = split_node(n, tok, cfg.recovery_cap) if tok.count(n.payload) > cfg.recovery_cap else [n]
        if any(tok.count(p.payload) > cfg.recovery_cap for p in pieces):
            raise ValueError(f"recovery source cannot fit cap {cfg.recovery_cap}: {n.node_id}")
        for p in pieces:
            if p is not n:
                nodes_by_id[p.node_id] = p
        work.extend(pieces)
    if independent:
        groups, kind, position = [[n] for n in work], "single", {}
    else:
        position = {nid: i for i, nid in enumerate(nodes_by_id)}

        def count(group):
            marks = unit_markers(ns, [position[x.node_id] for x in group], cfg)
            return tok.count(render_payload(group, marks))

        max_n = cfg.max_table_nodes if unit.kind == "table" else cfg.max_nodes
        groups = [g for g, _ in _greedy(work, count, cfg.recovery_cap, max_n)]
        if len(groups) == 1 and len(work) > 1 and [x.node_id for x in work] == list(unit.node_ids):
            groups = _halves(work, count)
        kind = "recovery"

    out = []
    for j, g in enumerate(groups, 1):
        # position is empty for independent (single-node) groups; unit_markers never reads the
        # index of a single node, so .get() is safe there and a multi-node group still needs it
        marks = unit_markers(ns, [position.get(x.node_id) for x in g], cfg)
        payload = render_payload(g, marks)
        span = _span_text(nodes_by_id, g[0].raw_start, g[-1].raw_end)
        out.append(Unit(
            unit_id=f"{unit.unit_id}.r{j}", doc_id=unit.doc_id, scope_id=unit.scope_id, kind=kind,
            node_ids=[x.node_id for x in g], markers=marks, payload=payload,
            source_tokens=tok.count(payload), raw_start=g[0].raw_start, raw_end=g[-1].raw_end,
            raw_sha256=_sha(span) if span is not None else "", split_from=unit.unit_id))
    return out


# ---- recovery policy (moved from engine/doc_job.py _DocJob._unit_flow/_recovery) -------------------
def should_recover(hard, cfg) -> bool:
    """One quality-recovery round when the first pass has hard failures (never for operational errors)."""
    return bool(hard and cfg.selective_recovery and cfg.quality_recovery_rounds >= 1
                and "operational_error" not in hard)


def recovery_mode(unit, hard):
    """(mode, independent): marker failures of a multi-node unit -> independent single-node calls,
    anything else -> regroup into <= recovery_cap units."""
    independent = is_marker_failure(hard) and len(unit.node_ids) > 1
    return ("independent" if independent else "regroup"), independent


def regroup_is_noop(unit, kids, independent: bool) -> bool:
    """Regroup returned the unit unchanged: single-node calls are the only smaller re-run."""
    return (not independent and len(unit.node_ids) > 1 and len(kids) == 1
            and list(kids[0].node_ids) == list(unit.node_ids))


def recovery_reason(mode: str, hard) -> str:
    return f"{mode}:" + ",".join(sorted({_code(h) for h in hard}))


def is_identical_retry(kids, builds, first_request) -> bool:
    """Greedy decoding would repeat the first pass: one child with the same prompt and budget."""
    return (len(kids) == 1 and len(builds) == 1 and builds[0].request.prompt == first_request.prompt
            and builds[0].request.max_new_tokens == first_request.max_new_tokens)

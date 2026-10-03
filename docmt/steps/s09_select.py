"""Step 9 - select: choose, per unit, between the first-pass output and the recovery output.

Input : the first-pass Attempt, and (when the unit went through recovery) the recovery Attempts,
        the recovery child units and the number of children that could not be built.
Output: the chosen candidate, per-node final texts, nodes left empty, the request ids whose
        outputs are used, and flags.

Rules (lam_docmt runtime, unchanged):
- candidate with FEWER hard failures wins; tie or worse keeps the first pass
  (flags recovery_selected / recovery_tie_kept_original / recovery_worse_kept_original);
- a node incomplete in the chosen candidate is filled from the other candidate when that one is
  complete there (node_filled_from_<name>), else the non-empty partial text is kept (node_partial),
  else it stays empty (unresolved_empty);
- a unit is accepted iff the chosen candidate has no hard failure and no node is empty.

Moved out of engine/doc_job.py (_DocJob._first_candidate/_recovery_candidate/_select).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _Candidate:
    name: str                    # "first" | "recovery"
    hard: list
    warnings: list
    texts: dict                  # unit-level node id -> (text, complete, [request_ids])
    attempts: list


def first_candidate(unit, a0) -> _Candidate:
    texts = {}
    for n in unit.node_ids:
        t = (a0.parsed.segments.get(n) or "").strip()
        texts[n] = (t, bool(t), [a0.request.request_id])
    return _Candidate("first", list(a0.check.hard), list(a0.check.warnings), texts, [a0])


def recovery_candidate(unit, attempts, children, n_unfit: int, nodes_by_id) -> _Candidate:
    """Union of the recovery children's segments; a node split into pieces (x.1, x.2 ...) is the
    pieces joined by " " in source order, complete only when every piece is non-empty."""
    segs, owner = {}, {}
    for a in attempts:
        for n, t in (a.parsed.segments or {}).items():
            segs[n] = (t or "").strip()
            owner[n] = a.request.request_id
    child_nodes = [n for c in children for n in c.node_ids]
    texts = {}
    for x in unit.node_ids:
        if x in segs:
            texts[x] = (segs[x], bool(segs[x]), [owner[x]])
            continue
        pieces = [n for n in child_nodes if n.startswith(x + ".")]
        pieces.sort(key=lambda n: (nodes_by_id[n].raw_start if n in nodes_by_id else 0))
        parts = [segs.get(p, "") for p in pieces]
        texts[x] = (" ".join(p for p in parts if p), bool(pieces) and all(parts),
                    [owner[p] for p in pieces if p in owner])
    hard = [h for a in attempts for h in a.check.hard] + ["prompt_too_long"] * n_unfit
    warns = [w for a in attempts for w in a.check.warnings]
    return _Candidate("recovery", hard, warns, texts, list(attempts))


def choose_candidate(c0: _Candidate, c1, skipped=None):
    """(chosen, flags). `skipped` = reason the recovery round was skipped (no c1 then)."""
    flags = []
    chosen = c0
    if skipped:
        flags.append(f"recovery_skipped_{skipped}")
    elif c1 is not None:
        if len(c1.hard) < len(c0.hard):
            chosen = c1
            flags.append("recovery_selected")
        elif len(c1.hard) == len(c0.hard):
            flags.append("recovery_tie_kept_original")
        else:
            flags.append("recovery_worse_kept_original")
    return chosen, flags


def fill_nodes(unit, chosen: _Candidate, other):
    """(final {node: text}, empty [node], used {request_id}, flags) for the unit's nodes."""
    final, empty, used, flags = {}, [], set(), []
    for x in unit.node_ids:
        t, complete, rids = chosen.texts[x]
        if complete:
            final[x] = t
            used.update(rids)
            continue
        alt = other.texts.get(x) if other is not None else None
        if alt and alt[1]:
            final[x] = alt[0]
            used.update(alt[2])
            flags.append(f"node_filled_from_{other.name}:{x}")
        elif t or (alt and alt[0]):
            src_t = (t, rids) if t else (alt[0], alt[2])
            final[x] = src_t[0]
            used.update(src_t[1])
            flags.append(f"node_partial:{x}")
        else:
            final[x] = ""
            empty.append(x)
            flags.append(f"unresolved_empty:{x}")
    return final, empty, used, flags


def is_accepted(chosen: _Candidate, empty) -> bool:
    return not chosen.hard and not empty

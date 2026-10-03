"""Step 10a - assemble: per-node translations -> one Markdown document on the raw source layout.

Input : parsed nodes (raw layout: lead / content_raw / trail), work nodes (after splitting long
        nodes into pieces nX.1, nX.2 ...), selected text per work node, and a function mapping a
        work node to its original node.
Output: the document text, the translation of every translatable original node, work nodes left
        empty, and how many translatable nodes had no work piece at all.

Rule (SPEC section 2, unchanged): out = "".join(lead + translation-or-copy + trail); a node split
into pieces is its pieces' non-empty translations joined by " " in raw order. Copy nodes
(numbers, separators, protected names) are written back verbatim.

Moved out of engine/doc_job.py (_DocJob.finish).
"""
from __future__ import annotations


def assemble(nodes, work, work_text: dict, orig_of):
    """(out_text, trans {node_id: text}, empty_work [ids], nodes_without_pieces)."""
    pieces = {n.node_id: [] for n in nodes}
    for w in work:
        if w.translatable:
            pieces[orig_of(w.node_id, w)].append(w)
    trans, empty_work, no_pieces = {}, [], 0
    for n in nodes:
        if not n.translatable:
            continue
        ps = sorted(pieces.get(n.node_id) or [], key=lambda w: w.raw_start)
        if not ps:
            empty_work.append(n.node_id)
            no_pieces += 1
        texts = [(w.node_id, work_text.get(w.node_id, "")) for w in ps]
        empty_work += [wid for wid, t in texts if not t]
        trans[n.node_id] = " ".join(t for _, t in texts if t)
    out_text = "".join(n.lead + (trans.get(n.node_id, "") if n.translatable else n.content_raw) + n.trail
                       for n in nodes)
    return out_text, trans, empty_work, no_pieces


def strip_marker_leaks(out_text: str, leak_re):
    """Final guard: internal markers never reach a prediction. (text, leaked)."""
    if leak_re.search(out_text):
        return leak_re.sub("", out_text), True
    return out_text, False

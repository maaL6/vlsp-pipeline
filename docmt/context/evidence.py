"""Split from lam_docmt/context.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .common import DEFAULTS, ELLIPSIS, KIND_OF_ORIGIN, _overlaps, _strip_markers, guess_lang
from ..core.datatypes import EvidenceItem

def _opt(cfg, key):
    extra = getattr(cfg, "extra", None) or {}
    return int(extra.get(key, DEFAULTS[key]))

def _unit_view(unit, nodes_by_id):
    nodes = [nodes_by_id[nid] for nid in unit.node_ids if nid in (nodes_by_id or {})]
    if nodes and len(nodes) == len(unit.node_ids):
        parts, starts, pos = [], [], 0
        for n in nodes:
            starts.append(pos)
            parts.append(n.payload or "")
            pos += len(n.payload or "") + 1
        spans = [(n.raw_start, n.raw_end) for n in nodes]
        return nodes, "\n".join(parts), starts, spans
    # Fallback (node ids missing from nodes_by_id): unit payload without markers, unit raw span.
    text = "\n".join(_strip_markers(unit.payload))
    return nodes, text, None, [(unit.raw_start, unit.raw_end)]

def _sentences(text):
    parts = re.split(r"(?<=[.;:!?])\s+(?=[\"“(A-ZĐÀ-Ỹ0-9])", text)
    return [p for p in parts if p.strip()]

def _tail_fit(parts, tok, cap, joiner="\n") -> str:
    chosen = []
    for p in reversed(parts):
        trial = [p] + chosen
        if tok.count(joiner.join(trial)) > cap:
            break
        chosen = trial
    return joiner.join(chosen)

def _history_text(entry, tok, cap) -> tuple:
    src, tgt = entry.get("src_segments") or [], entry.get("tgt_segments") or []
    if src and len(src) == len(tgt):
        text = _tail_fit([f"{s} => {t}" for s, t in zip(src, tgt)], tok, cap)
        if text:
            return text, "pairs"
    text = _tail_fit(tgt, tok, cap)
    if text:
        return text, "target"
    if tgt:
        text = _tail_fit(_sentences(tgt[-1]), tok, cap, joiner=" ")
        if text:
            return text, "target_sentences"
    return "", "none"

def _truncate_title(item, tok, limit):
    cur = tok.count(item.text)
    if cur <= limit:
        return item
    full = (item.ref or {}).get("full_tokens", cur)
    if limit <= 2:
        return replace(item, text="", ref=dict(item.ref, truncated=True, full_tokens=full))
    cut = tok.truncate_tokens(item.text, limit - 2)
    if " " in cut and not item.text[len(cut):len(cut) + 1].isspace():
        cut = cut[:cut.rfind(" ")]
    text = cut.rstrip(" ,;:-–") + " " + ELLIPSIS if cut.strip() else ""
    while text and tok.count(text) > limit:
        cut = cut[:cut.rfind(" ")] if " " in cut else ""
        text = cut.rstrip(" ,;:-–") + " " + ELLIPSIS if cut.strip() else ""
    return replace(item, text=text, ref=dict(item.ref, truncated=True, full_tokens=full))

def _fit(cands, tok, cap, title_cap):
    kept, dropped = [], []
    if cap <= 0:
        return [], list(cands)
    for it in cands:
        if it.origin == "source_title":
            t = _truncate_title(it, tok, min(title_cap, cap))
            while t.text and tok.count(render_background(kept + [t])) > cap:
                excess = tok.count(render_background(kept + [t])) - cap
                t = _truncate_title(t, tok, max(0, tok.count(t.text) - excess - 1))
            if t.text:
                kept.append(t)
            else:
                dropped.append(it)
            continue
        if tok.count(render_background(kept + [it])) <= cap:
            kept.append(it)
        else:
            dropped.append(it)
    for it in kept:
        it.tokens = tok.count(it.text)
    for it in dropped:
        it.tokens = tok.count(it.text)
    return kept, dropped

def select_evidence_detailed(unit, nodes_by_id, index, memory, glossary, tok, cfg, doc_id, src_lang=None) -> tuple:
    """(kept, dropped, info). kept is what select_evidence returns; dropped/info are for traces."""
    unit_nodes, text, starts, spans = _unit_view(unit, nodes_by_id)
    scope = unit.scope_id or (unit_nodes[0].scope_id if unit_nodes else "main")
    cands = []
    if getattr(cfg, "source_context", False) and index is not None:
        index.check_doc(doc_id, unit)
        t = index.title_item(scope, spans)
        if t is not None:
            cands.append(t)
        p = index.path_item(unit_nodes, spans)
        if p is not None:
            cands.append(p)
        cands += index.definition_items(text, scope, spans, _opt(cfg, "context_max_definitions"))
        cands += index.reference_items(text, starts, unit_nodes, scope, spans,
                                       _opt(cfg, "context_max_references"))
    if getattr(cfg, "soft_glossary", False) and glossary is not None:
        lang = src_lang or (index.src_lang if index is not None else guess_lang(text))
        for s, tg, e in glossary.match_entries(text, lang, getattr(cfg, "soft_terms", 8)):
            cands.append(EvidenceItem("glossary", "train_glossary", f"{s} translates to {tg}", 0,
                                      ref={"src": s, "tgt": tg, "domain": e.get("domain"),
                                           "audit_status": e.get("audit_status"),
                                           "line": e.get("line")}))
    recovery = bool(getattr(unit, "split_from", None)) or unit.kind == "recovery"
    if getattr(cfg, "target_history", False) and memory is not None and not recovery:
        cap = _opt(cfg, "context_history_item_cap")
        for e in memory.recent(doc_id, scope, getattr(cfg, "recent_accepted_units", 2)):
            if e.get("doc_id") != doc_id or e.get("scope_id") != scope:
                continue
            uid = str(e.get("unit_id"))
            if uid == unit.unit_id or uid.startswith(unit.unit_id + "."):
                continue
            if e.get("raw_span") and _overlaps([tuple(e["raw_span"])], spans):
                continue
            htext, form = _history_text(e, tok, cap)
            if htext:
                cands.append(EvidenceItem("history", "accepted_history", htext, 0,
                                          ref={"unit_id": e.get("unit_id"), "scope_id": scope,
                                               "seq": e.get("seq"), "form": form}))
    cap = int(getattr(cfg, "evidence_cap", 512))
    kept, dropped = _fit(cands, tok, cap, _opt(cfg, "context_title_cap"))
    info = {"candidates": len(cands), "kept": len(kept), "dropped": len(dropped),
            "background_tokens": tok.count(render_background(kept)) if kept else 0,
            "evidence_cap": cap}
    return kept, dropped, info

def select_evidence(unit, nodes_by_id, index, memory, glossary, tok, cfg, doc_id, src_lang=None) -> list:
    """Evidence items for one unit, in priority order, rendered background <= cfg.evidence_cap."""
    return select_evidence_detailed(unit, nodes_by_id, index, memory, glossary, tok, cfg, doc_id, src_lang)[0]

_HEADERS = {
    "definition": "Definitions given in this document:",
    "reference": "Articles of this document referred to in the text:",
    "glossary": "Reference translations:",       # same wording as prompts.build_user
    "history": "Previously accepted translations from the same part of this document:",
}

def render_background(items) -> str:
    """Background block for prompts.HY_BACKGROUND, in the given (priority) order. Glossary lines use
    the A1-native "X translates to Y" form; pass terms=() to build_prompt when using this text."""
    lines, prev = [], None
    for it in items:
        if not it.text:
            continue
        if it.origin == "source_title":
            label = "Document title" if (it.ref or {}).get("scope", "main") == "main" else \
                "Title of this part of the document"
            lines.append(f"{label}: {it.text}")
            group = "title"
        elif it.origin == "source_path":
            lines.append(f"Current position: {it.text}")
            group = "path"
        else:
            group = KIND_OF_ORIGIN.get(it.origin, it.kind)
            if group != prev:
                lines.append(_HEADERS.get(group, group.capitalize() + ":"))
            lines.append(f"- {it.text}" if group in ("definition", "reference") else it.text)
        prev = group
    return "\n".join(lines)

def split_terms(items) -> tuple:
    """(non-glossary items, [(src, tgt)]) for callers that prefer prompts' own terms block."""
    rest = [it for it in items if it.origin != "train_glossary"]
    terms = [(it.ref.get("src"), it.ref.get("tgt")) for it in items if it.origin == "train_glossary"]
    return rest, terms

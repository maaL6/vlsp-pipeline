"""Intermediate results of steps s01..s05 without a model (python -m docmt inspect --stop-after STAGE).

Runs the same preparation as a real run (_DocJob.prepare: parse -> split -> pack -> budget split)
and writes one JSONL record per item of the requested stage, so a single step can be changed and its
output compared before any GPU run:

  parse    one record per parsed node (kind, scope, offsets, raw text, flags, protected labels)
  payload  one record per translatable node: model input text (normalize(unwrap(raw))) + tokens
  split    one record per work node after long-node splitting (pieces nX.1 ...)
  pack     one record per translation unit (node ids, markers, source tokens, payload)
  prompt   one record per unit: first-pass prompt, template, prompt/max_new tokens, evidence
           (history evidence is empty here: no accepted outputs exist without a model)

Only the source side is read. Nothing is generated.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ..core.hashing import sha256_text
from ..tracing.writer import TraceWriter
from .doc_job import _as_components, _DocJob
from .run import _Run

STAGES = ("parse", "payload", "split", "pack", "prompt")


def _node(n, tok, full=True) -> dict:
    d = asdict(n)
    if not full:
        d = {k: d[k] for k in ("node_id", "kind", "scope_id", "translatable", "payload", "flags")}
    d["payload_tokens"] = tok.count(n.payload) if n.payload else 0
    return d


def inspect_items(items, cfg, tok, stage: str, out_path: Path, components=None) -> dict:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    comp = _as_components(components)
    run = _Run(cfg, tok, None, comp, TraceWriter(None), None, cfg.run_id or "inspect")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"stage": stage, "documents": 0, "failed": [], "records": 0, "kinds": Counter()}
    with open(out_path, "w", encoding="utf-8") as f:
        def emit(item, rec):
            rec = {"direction": item.direction, "doc_id": item.sample_id, **rec}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            stats["records"] += 1

        for item in items:
            job = _DocJob(run, item)
            try:
                job.prepare()
            except Exception as exc:
                stats["failed"].append({"doc_id": item.sample_id, "direction": item.direction, "error": repr(exc)})
                continue
            stats["documents"] += 1
            if stage == "parse":
                for n in job.nodes:
                    stats["kinds"][n.kind] += 1
                    emit(item, _node(n, tok))
            elif stage == "payload":
                for n in job.nodes:
                    if n.translatable:
                        stats["kinds"][n.kind] += 1
                        emit(item, _node(n, tok, full=False))
            elif stage == "split":
                for n in job.work:
                    stats["kinds"][n.kind] += 1
                    emit(item, {**_node(n, tok, full=False), "raw_start": n.raw_start, "raw_end": n.raw_end,
                                "is_piece": "." in n.node_id})
            elif stage == "pack":
                for u in job.units:
                    stats["kinds"][u.kind] += 1
                    emit(item, {"unit_id": u.unit_id, "kind": u.kind, "scope_id": u.scope_id,
                                "node_ids": list(u.node_ids), "markers": list(u.markers),
                                "source_tokens": u.source_tokens, "split_from": u.split_from,
                                "payload": u.payload})
            else:
                for u in job.units:
                    stats["kinds"][u.kind] += 1
                    b = job._build(u, 0, allow_history=False)
                    rec = {"unit_id": u.unit_id, "kind": u.kind, "source_tokens": u.source_tokens, "fits": b is not None}
                    if b is not None:
                        rec.update(template=b.template, prompt_tokens=b.prompt_tokens,
                                   max_new_tokens=b.request.max_new_tokens,
                                   total_tokens=b.prompt_tokens + b.request.max_new_tokens + cfg.margin,
                                   prompt_sha256=sha256_text(b.request.prompt), prompt=b.request.prompt,
                                   evidence=[{"kind": e.kind, "origin": e.origin, "tokens": e.tokens, "text": e.text}
                                             for e in b.items],
                                   evidence_dropped=[{"kind": e.kind, "tokens": e.tokens} for e in b.dropped])
                    emit(item, rec)
    stats["kinds"] = dict(stats["kinds"])
    stats["out"] = str(out_path)
    return stats

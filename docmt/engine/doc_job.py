"""Document-level runtime (SPEC section 3; draft section 2 "Sinh và phục hồi").

Per document: read raw source -> parse -> split long nodes / pack units -> (SourceIndex) ->
for each unit an evidence-fitted prompt (drop evidence items, lowest priority first, until
prompt + budget + margin <= max_sequence; a unit whose source alone does not fit is split) ->
generate -> parse_output -> check_unit -> at most ONE quality-recovery round (marker failures ->
independent single-node calls; other hard failures -> regroup <= recovery_cap tokens; recovery
prompts never carry target history) -> candidate selection (fewer hard failures; tie keeps the
first pass + flag) -> accepted units -> Memory -> assemble on the raw source layout ->
check_document -> prediction + logs.

Scheduling: up to cfg.batch_size requests in flight across documents; strictly one request at a
time inside a document when source context or target history is on. Operational problems (network, OOM, file)
are counted separately from quality recovery. Internal markers never reach a prediction.
Reference/gold text is never read here: only the source side of a doc-direction is opened.
"""
# Split from lam_docmt/runtime.py (statements verbatim; tools/split_from_lam_docmt.py).
from __future__ import annotations

import bisect
import dataclasses
import importlib
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

from ..core.codes import _code, _dedupe
from ..core.config import RunConfig
from ..core.datatypes import Attempt, DocumentResult, GenRequest, Node, ParseResult, UnitCheck
from ..core.hashing import sha256_text
from .scheduler import _parallel
from ..inputs.sources import read_source
from ..steps.s04_prompt import render_unit_prompt
from ..steps.s05_budget import fit_prompt, fits_alone, recovery_source_tokens
from ..steps.s06_parse_output import leak_pattern, strip_internal_markers
from ..steps.s07_validate_unit import combine_hard
from ..steps.s08_recovery import (is_identical_retry, recovery_mode, recovery_reason, regroup_is_noop,
                                  should_recover)
from ..steps.s09_select import choose_candidate, fill_nodes, first_candidate, is_accepted, recovery_candidate
from ..steps.s10_assemble import assemble, strip_marker_leaks
from ..steps.s10_doc_check import decide_status
from ..tracing.records import make_trace_record

# -- collaborators -----------------------------------------------------------------------------------
_SOURCES = {
    "parse_document": ("docmt.steps.s01_parse", "parse_document"),
    "split_long_nodes": ("docmt.steps.s03_split", "split_long_nodes"),
    "pack_units": ("docmt.steps.s03_pack", "pack_units"),
    "render_payload": ("docmt.steps.s03_pack", "render_payload"),
    "parse_output": ("docmt.steps.s06_parse_output", "parse_output"),
    "output_budget": ("docmt.steps.s05_budget", "output_budget"),
    "recovery_units": ("docmt.steps.s08_recovery", "recovery_units"),
    "marker_namespace": ("docmt.steps.s03_pack", "marker_namespace"),
    "check_unit": ("docmt.steps.s07_validate_unit", "check_unit"),
    "check_document": ("docmt.steps.s10_doc_check", "check_document"),
    "SourceIndex": ("docmt.context.source_index", "SourceIndex"),
    "Memory": ("docmt.context.memory", "Memory"),
    "Glossary": ("docmt.context.glossary", "Glossary"),
    "select_evidence": ("docmt.context.evidence", "select_evidence"),
    "render_background": ("docmt.context.evidence", "render_background"),
}

class Components:
    """Collaborators of the runtime. Anything not passed is imported lazily from the real module
    (parse/units/validate/context) on first use, so tests can inject fakes."""

    NAMES = tuple(_SOURCES)

    def __init__(self, **given):
        unknown = sorted(set(given) - set(_SOURCES))
        if unknown:
            raise TypeError(f"unknown components {unknown}")
        self._given = {k: v for k, v in given.items() if v is not None}
        self._loaded = {}

    def __getattr__(self, name):
        if name.startswith("_") or name not in _SOURCES:
            raise AttributeError(name)
        if name in self._given:
            return self._given[name]
        if name not in self._loaded:
            mod, attr = _SOURCES[name]
            self._loaded[name] = getattr(importlib.import_module(mod), attr)
        return self._loaded[name]

    def available(self, name) -> bool:
        try:
            getattr(self, name)
            return True
        except (ImportError, AttributeError):
            return False

    def origin(self) -> dict:
        return {k: ("injected" if k in self._given else _SOURCES[k][0]) for k in _SOURCES}

def _as_components(components) -> Components:
    if isinstance(components, Components):
        return components
    return Components(**(components or {}))

def _as_config(config) -> RunConfig:
    if isinstance(config, RunConfig):
        return config
    if isinstance(config, dict):
        return RunConfig().apply(config)
    raise TypeError("config must be a RunConfig or a dict")

# -- per-request bookkeeping -------------------------------------------------------------------------
@dataclass
class _Built:
    unit: Unit
    request: GenRequest
    items: list
    dropped: list
    prompt_tokens: int
    attempt: int
    parent_id: Optional[str]
    reason: Optional[str]
    template: str

@dataclass
class _Recovery:
    mode: str
    attempts: list = field(default_factory=list)
    builds: list = field(default_factory=list)
    children: list = field(default_factory=list)
    unfit: list = field(default_factory=list)
    skipped: Optional[str] = None

@dataclass
class _UnitOutcome:
    unit: Unit
    final: dict                  # unit-level node id -> text ("" when unresolved_empty)
    accepted: bool
    hard: list
    warnings: list
    flags: list
    chosen: str
    recovery_mode: Optional[str]
    requests: int
    recovery_requests: int
    empty_nodes: list
    op_error: bool = False

# -- one document --------------------------------------------------------------------------------------
class _DocJob:
    def __init__(self, run: "_Run", item: DocItem):
        self.run, self.item = run, item
        self.cfg, self.tok, self.comp = run.cfg, run.tok, run.comp
        self.doc_id = item.sample_id
        self.t0 = time.time()
        self.src = ""
        self.nodes: list = []
        self.work: list = []
        self.units: list = []
        self.nodes_by_id: dict = {}
        self.ns = "SEG"
        self.leak_re = None
        self.index = None
        self.memory = None
        self.split_parent: dict = {}
        self.uncovered: list = []
        self.roundtrip = {"ok": False, "offsets_ok": False}
        self.validator_ok = True
        self.flags = Counter()
        self.counts = Counter()
        self.evidence_counts = Counter()
        self.result: Optional[DocumentResult] = None
        self._starts: list = []

    # ---- preparation -----------------------------------------------------------------------------
    def prepare(self):
        cfg, comp, tok = self.cfg, self.comp, self.tok
        self.src = src = read_source(self.item.source_path)
        if not src.strip():
            raise _DocFailure("empty_document")
        self.nodes = list(comp.parse_document(src, self.doc_id, self.item.src_lang, tok))
        self._starts = [n.raw_start for n in self.nodes]
        self._check_roundtrip()
        self.ns = comp.marker_namespace(src)
        self.leak_re = leak_pattern(self.ns)
        self.work = list(comp.split_long_nodes(self.nodes, tok, cfg))
        self.nodes_by_id = {n.node_id: n for n in self.nodes}
        self.nodes_by_id.update({n.node_id: n for n in self.work})
        units = list(comp.pack_units(self.work, self.doc_id, tok, cfg, self.ns))
        self._check_coverage(units)
        self.units = self._fit_units(units)
        if cfg.source_context:
            self.index = comp.SourceIndex.build(self.nodes, self.item.src_lang)
        if cfg.target_history:
            self.memory = comp.Memory()

    def _check_roundtrip(self):
        src, nodes = self.src, self.nodes
        ok = "".join(n.raw for n in nodes) == src
        pos, offsets_ok = 0, True
        for n in nodes:
            if n.raw_start != pos or src[n.raw_start:n.raw_end] != n.raw:
                offsets_ok = False
                break
            pos = n.raw_end
        offsets_ok = offsets_ok and pos == len(src)
        self.roundtrip = {"ok": ok, "offsets_ok": offsets_ok}
        if not (ok and offsets_ok):
            raise _DocFailure("parse_roundtrip_failed")

    def _check_coverage(self, units):
        seen = Counter(n for u in units for n in u.node_ids)
        unknown = [n for n in seen if n not in self.nodes_by_id]
        if unknown:
            raise _DocFailure(f"unit_references_unknown_nodes:{unknown[:5]}")
        dup = [n for n, k in seen.items() if k > 1]
        if dup:
            raise _DocFailure(f"node_in_several_units:{dup[:5]}")
        self.uncovered = [n.node_id for n in self.work if n.translatable and n.node_id not in seen]

    def _fits_alone(self, u) -> bool:
        return fits_alone(u, self.item.tgt_lang, self.tok, self.cfg, self.comp.output_budget)

    def _fit_units(self, units):
        """A unit whose source alone (no evidence) cannot fit with its budget is regrouped into
        <= recovery_cap pieces before the first pass (draft: giảm evidence trước, chia nhỏ nguồn)."""
        out = []
        for u in units:
            if not u.node_ids or self._fits_alone(u):
                out.append(u)
                continue
            try:
                kids = list(self.comp.recovery_units(u, self.nodes_by_id, self.tok, self.cfg, self.ns, independent=False))
                self._register(kids)
            except Exception as exc:
                self.run.log_error(self, {"kind": "budget_split_error", "unit_id": u.unit_id, "error": repr(exc)})
                kids = []
            if not kids or (len(kids) == 1 and list(kids[0].node_ids) == list(u.node_ids)):
                out.append(u)  # will be reported as prompt_too_long
                continue
            self.flags["budget_split"] += 1
            for k, c in enumerate(kids, 1):
                c2 = dataclasses.replace(c, unit_id=f"{u.unit_id}.s{k}", kind=u.kind, split_from=u.unit_id)
                self.split_parent[c2.unit_id] = u.unit_id
                out.append(c2)
        return out

    def _register(self, units):
        """Make sure every node id of (recovery) units is known; regenerate split pieces if the
        units module did not register them."""
        for u in units:
            for nid in u.node_ids:
                if nid in self.nodes_by_id:
                    continue
                base = nid
                while "." in base and base not in self.nodes_by_id:
                    base = base.rsplit(".", 1)[0]
                if base in self.nodes_by_id:
                    for p in self.comp.split_long_nodes([self.nodes_by_id[base]], self.tok, self.cfg.recovery_cap):
                        self.nodes_by_id.setdefault(p.node_id, p)
                if nid not in self.nodes_by_id and len(u.node_ids) == 1:
                    b = self.nodes_by_id.get(base)
                    self.nodes_by_id[nid] = Node(
                        node_id=nid, kind=b.kind if b else "paragraph", raw_start=u.raw_start, raw_end=u.raw_end,
                        lead="", content_raw=self.src[u.raw_start:u.raw_end], trail="", payload=u.payload,
                        translatable=True, scope_id=u.scope_id, path=list(b.path) if b else [],
                        protected=dict(b.protected) if b else {}, table=b.table if b else None,
                        flags=list(b.flags if b else []) + ["synthesized_piece"])
                if nid not in self.nodes_by_id:
                    raise _DocFailure(f"unknown_piece_node:{nid}")

    # ---- prompts ---------------------------------------------------------------------------------
    def _evidence(self, unit, allow_history: bool) -> list:
        cfg = self.cfg
        hist = cfg.target_history and allow_history
        if not (cfg.source_context or cfg.soft_glossary or hist):
            return []
        items = self.comp.select_evidence(unit, self.nodes_by_id, self.index, self.memory if hist else None,
                                          self.run.glossary if cfg.soft_glossary else None, self.tok, cfg,
                                          self.doc_id)
        keep = []
        for it in items or []:
            if it.kind == "glossary":
                ok = cfg.soft_glossary
            elif it.kind == "history":
                ok = hist and it.origin == "accepted_history"
            else:
                ok = cfg.source_context
            if ok:
                keep.append(it)
        return keep

    def _render(self, unit, items):
        render_bg = self.comp.render_background if items else None
        return render_unit_prompt(unit, items, self.item.src_lang, self.item.tgt_lang, self.cfg, render_bg)

    def _build(self, unit, attempt: int, allow_history: bool, parent_id=None, reason=None) -> Optional[_Built]:
        cfg, tok = self.cfg, self.tok
        if attempt == 1:
            ok, n_src = recovery_source_tokens(unit, tok, cfg)
            if not ok:
                self.run.log_error(self, {"kind": "recovery_source_cap_exceeded", "unit_id": unit.unit_id,
                                          "source_tokens": n_src, "cap": cfg.recovery_cap})
                return None
        fitted = fit_prompt(unit, self._evidence(unit, allow_history), self._render, tok, cfg,
                            self.comp.output_budget)
        if fitted is None:
            return None
        prompt, template, ptoks, budget, items, dropped = fitted
        rid = f"{self.run.run_id}|{self.item.direction}|{unit.unit_id}|a{attempt}"
        meta = {"run_id": self.run.run_id, "direction": self.item.direction, "doc_id": self.doc_id,
                "scope_id": unit.scope_id, "unit_id": unit.unit_id, "unit_kind": unit.kind, "attempt": attempt,
                "parent_unit_id": parent_id, "node_ids": list(unit.node_ids), "markers": list(unit.markers),
                "payload": unit.payload}
        req = GenRequest(request_id=rid, prompt=prompt, max_new_tokens=budget,
                         repetition_penalty=cfg.repetition_penalty, meta=meta)
        self.evidence_counts["items"] += len(items)
        self.evidence_counts["dropped"] += len(dropped)
        return _Built(unit, req, items, dropped, ptoks, attempt, parent_id, reason, template)

    # ---- evaluation ------------------------------------------------------------------------------
    def _evaluate(self, b: _Built, res: GenResult) -> Attempt:
        unit = b.unit
        if res.stop_type == "error" or res.error:
            parsed = ParseResult(segments={}, hard=["operational_error"])
            check = UnitCheck(hard=["operational_error"], warnings=[], details={"error": res.error})
            return Attempt(unit, b.request, res, parsed, check, b.items, b.attempt, b.reason)
        try:
            parsed = self.comp.parse_output(unit, res.text)
        except Exception as exc:  # units bug on odd model output: treat as a hard failure of this attempt
            self.run.log_error(self, {"kind": "parse_output_error", "unit_id": unit.unit_id, "error": repr(exc)})
            parsed = ParseResult(segments={}, hard=["parse_output_error"])
        segs, leaks = strip_internal_markers(parsed.segments, self.leak_re)
        parsed = ParseResult(segments=segs, hard=list(parsed.hard), extra=dict(parsed.extra or {}))
        try:
            chk = self.comp.check_unit(unit, self.nodes_by_id, res, parsed, self.item.src_lang, self.item.tgt_lang, self.cfg)
            hard_v, warn_v, details = list(chk.hard), list(chk.warnings), dict(chk.details or {})
        except Exception as exc:  # validator bug: keep going, mark checks as not done
            self.validator_ok = False
            self.run.log_error(self, {"kind": "validator_error", "unit_id": unit.unit_id, "error": repr(exc)})
            hard_v, warn_v, details = [], ["validator_error"], {"error": repr(exc)}
        hard = combine_hard(parsed.hard, hard_v, res.eos, leaks)
        check = UnitCheck(hard=hard, warnings=_dedupe(warn_v), details=details)
        return Attempt(unit, b.request, res, parsed, check, b.items, b.attempt, b.reason)

    # ---- unit flow (generator) -----------------------------------------------------------------
    def _unit_flow(self, unit):
        b0 = self._build(unit, 0, allow_history=True, parent_id=self.split_parent.get(unit.unit_id),
                         reason="budget_split" if unit.unit_id in self.split_parent else None)
        if b0 is None:
            return self._unfittable(unit)
        (r0,) = yield [b0.request]
        a0 = self._evaluate(b0, r0)
        rec = None
        if should_recover(a0.check.hard, self.cfg):
            rec = yield from self._recovery(unit, a0, b0)
        return self._select(unit, a0, b0, rec)

    def _recovery(self, unit, a0, b0):
        hard = a0.check.hard
        mode, independent = recovery_mode(unit, hard)
        try:
            kids = list(self.comp.recovery_units(unit, self.nodes_by_id, self.tok, self.cfg, self.ns,
                                                 independent=independent))
            self._register(kids)
            if regroup_is_noop(unit, kids, independent):
                kids = list(self.comp.recovery_units(unit, self.nodes_by_id, self.tok, self.cfg, self.ns,
                                                     independent=True))
                self._register(kids)
                mode = "regroup_noop->independent"
        except _DocFailure:
            raise
        except Exception as exc:
            self.run.log_error(self, {"kind": "recovery_units_error", "unit_id": unit.unit_id, "error": repr(exc)})
            self.flags["recovery_error"] += 1
            return _Recovery(mode=mode, skipped="recovery_units_error")
        reason = recovery_reason(mode, hard)
        rec = _Recovery(mode=mode, children=kids)
        for c in kids:
            b = self._build(c, 1, allow_history=False, parent_id=unit.unit_id, reason=reason)
            if b is None:
                rec.unfit.append(c)
            else:
                rec.builds.append(b)
        if is_identical_retry(kids, rec.builds, b0.request):
            rec.skipped = "identical_prompt"   # greedy decoding would repeat the first pass
            rec.builds = []
            return rec
        if rec.builds:
            results = yield [b.request for b in rec.builds]
            rec.attempts = [self._evaluate(b, r) for b, r in zip(rec.builds, results)]
        return rec

    def _unfittable(self, unit) -> _UnitOutcome:
        self.flags["prompt_too_long"] += 1
        return _UnitOutcome(unit=unit, final={n: "" for n in unit.node_ids}, accepted=False,
                            hard=["prompt_too_long"], warnings=[], flags=["prompt_too_long"], chosen="none",
                            recovery_mode=None, requests=0, recovery_requests=0, empty_nodes=list(unit.node_ids))

    # ---- selection (step 9: steps/s09_select.py) --------------------------------------------------
    def _select(self, unit, a0, b0, rec: Optional[_Recovery]) -> _UnitOutcome:
        c0 = first_candidate(unit, a0)
        c1 = None
        if rec is not None and not rec.skipped:
            c1 = recovery_candidate(unit, rec.attempts, rec.children, len(rec.unfit), self.nodes_by_id)
        chosen, flags = choose_candidate(c0, c1, rec.skipped if rec is not None else None)
        other = c1 if chosen is c0 else c0
        final, empty, used, node_flags = fill_nodes(unit, chosen, other)
        flags += node_flags
        accepted = is_accepted(chosen, empty)
        if accepted and self.memory is not None:
            src_text = "\n".join(self.nodes_by_id[x].payload for x in unit.node_ids)
            tgt_text = "\n".join(final[x] for x in unit.node_ids)
            self.memory.add(self.doc_id, unit.scope_id, unit.unit_id, src_text, tgt_text, True)
        # trace: one record per backend request of this unit
        rec_attempts = list(rec.attempts) if rec is not None else []
        builds = [b0] + (list(rec.builds) if rec is not None else [])
        for a, b in zip([a0] + rec_attempts, builds):
            self._trace(a, b, a.request.request_id in used)
        op_error = "operational_error" in chosen.hard or any(
            "operational_error" in a.check.hard for a in [a0] + rec_attempts if a.request.request_id in used)
        for f in flags:
            self.flags[_code(f)] += 1
        return _UnitOutcome(unit=unit, final=final, accepted=accepted, hard=list(chosen.hard),
                            warnings=list(chosen.warnings), flags=flags, chosen=chosen.name,
                            recovery_mode=rec.mode if rec is not None else None,
                            requests=1 + len(rec_attempts), recovery_requests=len(rec_attempts),
                            empty_nodes=empty, op_error=op_error)

    def _trace(self, a: Attempt, b: _Built, selected: bool):
        run, unit, req, res = self.run, a.unit, a.request, a.result
        sha = sha256_text(req.prompt)
        offsets = []
        for n in unit.node_ids:
            node = self.nodes_by_id.get(n)
            offsets.append([node.raw_start, node.raw_end] if node else None)
        rec = make_trace_record(
            run_id=run.run_id, direction=self.item.direction, doc_id=self.doc_id, scope_id=unit.scope_id,
            unit_id=unit.unit_id, unit_kind=unit.kind, attempt=a.attempt, parent_unit_id=b.parent_id,
            node_ids=list(unit.node_ids), source_offsets=offsets, raw_source_sha256=unit.raw_sha256,
            normalized_source=unit.payload,
            evidence=[{"kind": e.kind, "origin": e.origin, "text": e.text, "tokens": e.tokens} for e in b.items],
            prompt_sha256=sha, prompt_style=b.template, source_tokens=unit.source_tokens,
            input_tokens=res.input_tokens, output_tokens=res.output_tokens, max_new_tokens=req.max_new_tokens,
            eos=res.eos, stop_type=res.stop_type, output=res.text, segments=dict(a.parsed.segments),
            hard_failures=list(a.check.hard), warnings=list(a.check.warnings), retry_reason=a.retry_reason,
            selected=bool(selected), seconds=res.seconds, backend=res.backend or run.backend_name,
            request_id=req.request_id, cached=req.request_id in run.cached_ids,
            op_retries=run.op_retries.get(req.request_id, 0), error=res.error,
            evidence_dropped=[{"kind": e.kind, "origin": e.origin, "tokens": e.tokens} for e in b.dropped],
            prompt_tokens=b.prompt_tokens, unit_raw_span=[unit.raw_start, unit.raw_end],
            markers=list(unit.markers))
        run.writer.prompt(sha, req.prompt)
        run.writer.trace(rec)

    # ---- document flow (generator) -----------------------------------------------------------
    def flow(self):
        self.prepare()
        if self.cfg.target_history or self.cfg.source_context:
            outcomes = []
            for u in self.units:  # Context branches follow source order, including source-only context.
                out = yield from self._unit_flow(u)
                outcomes.append(out)
        else:
            outcomes = yield from _parallel([self._unit_flow(u) for u in self.units])
        return self.finish(outcomes)

    def finish(self, outcomes) -> DocumentResult:
        item, cfg = self.item, self.cfg
        work_text = {}
        for o in outcomes:
            work_text.update(o.final)
        out_text, trans, empty_work, no_pieces = assemble(self.nodes, self.work, work_text, self._orig_of)
        if no_pieces:
            self.flags["node_without_pieces"] += no_pieces
        doc_hard = []
        out_text, leaked = strip_marker_leaks(out_text, self.leak_re)
        if leaked:
            doc_hard.append("internal_marker_leak")
        unresolved_units = [o for o in outcomes if not o.accepted]
        unresolved_nodes = sorted({self._orig_of(x) for o in unresolved_units for x in o.unit.node_ids}
                                  | {self._orig_of(x) for x in empty_work})
        try:
            doc_check = self.comp.check_document(self.src, out_text, self.nodes, trans, unresolved_nodes, self.ns,
                                                 item.src_lang, item.tgt_lang)
            doc_check = dict(doc_check or {})
        except Exception as exc:
            self.validator_ok = False
            self.run.log_error(self, {"kind": "check_document_error", "error": repr(exc)})
            doc_check = {"error": repr(exc)}
        doc_hard += [str(h) for h in (doc_check.get("hard") or doc_check.get("hard_failures") or [])]
        doc_hard = _dedupe(doc_hard)
        op_error = any(o.op_error for o in outcomes)
        status, error = decide_status(op_error, out_text, unresolved_units, empty_work, self.uncovered, doc_hard)
        out_path = item.output_path
        if out_path is not None:
            out_path = Path(out_path)
            if status == "failed":
                if out_path.exists():
                    out_path.unlink()  # never leave a stale prediction behind a failed document
            else:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8", newline="") as f:
                    f.write(out_text)
        sel_hard = [h for o in outcomes for h in o.hard]
        sel_warn = [w for o in outcomes for w in o.warnings]
        record = self._status_record(status, error, outcomes, out_text if status != "failed" else None,
                                     doc_check, doc_hard, sel_hard, sel_warn, empty_work)
        self.run.writer.status(record)
        self.run.on_doc_done(self, record)
        self.result = DocumentResult(output_path=str(out_path) if out_path else "", status=status,
                                     diagnostics=record)
        return self.result

    def _orig_of(self, nid: str, node=None) -> str:
        node = node or self.nodes_by_id.get(nid)
        if node is None:
            return nid
        i = bisect.bisect_right(self._starts, node.raw_start) - 1
        return self.nodes[max(i, 0)].node_id

    def _status_record(self, status, error, outcomes, out_text, doc_check, doc_hard, sel_hard, sel_warn,
                       empty_work) -> dict:
        item, run = self.item, self.run
        kinds = Counter(n.kind for n in self.nodes)
        node_flags = Counter(f for n in self.nodes for f in (n.flags or []))
        reqs = sum(o.requests for o in outcomes)
        return {
            "run_id": run.run_id, "direction": item.direction, "doc_id": self.doc_id,
            "sample_id": item.sample_id, "category": item.category, "status": status, "error": error,
            "output_path": item.rel_output if out_text is not None else "",
            "output_sha256": sha256_text(out_text) if out_text is not None else None,
            "output_chars": len(out_text) if out_text is not None else 0,
            "source_path": item.rel_source, "source_sha256": sha256_text(self.src), "source_chars": len(self.src),
            "roundtrip": dict(self.roundtrip), "marker_namespace": self.ns,
            "nodes": len(self.nodes), "translatable_nodes": sum(1 for n in self.nodes if n.translatable),
            "work_nodes": len(self.work), "node_kinds": dict(kinds), "node_flags": dict(node_flags),
            "scopes": dict(Counter(n.scope_id for n in self.nodes)),
            "signature_name_nodes": kinds.get("signature_name", 0), "signature_nodes": kinds.get("signature", 0),
            "table_nodes": kinds.get("table_cell", 0), "broken_table_nodes": node_flags.get("broken_table", 0),
            "units": len(self.units), "unit_kinds": dict(Counter(u.kind for u in self.units)),
            "requests": reqs, "recovery_requests": sum(o.recovery_requests for o in outcomes),
            "recovery_units": sum(1 for o in outcomes if o.recovery_mode is not None),
            "recovery_modes": dict(Counter(o.recovery_mode for o in outcomes if o.recovery_mode)),
            "recovered_units": sum(1 for o in outcomes if o.chosen == "recovery" and o.accepted),
            "accepted_units": sum(1 for o in outcomes if o.accepted),
            "unresolved_units": [{"unit_id": o.unit.unit_id, "hard": o.hard, "chosen": o.chosen}
                                 for o in outcomes if not o.accepted],
            "unresolved_empty": list(empty_work), "uncovered_nodes": list(self.uncovered),
            "selected_hard_failures": sel_hard, "selected_warnings": sel_warn,
            "doc_hard_failures": doc_hard, "doc_check": doc_check, "flags": dict(self.flags),
            "evidence": dict(self.evidence_counts), "validator_available": self.validator_ok,
            "op_retries": sum(run.op_retries.get(r, 0) for r in self._request_ids(outcomes)),
            "op_errors": sum(1 for o in outcomes if o.op_error),
            "cache_hits": sum(1 for r in self._request_ids(outcomes) if r in run.cached_ids),
            "seconds": round(time.time() - self.t0, 3), "backend": run.backend_name,
        }

    def _request_ids(self, outcomes):
        prefix = f"{self.run.run_id}|{self.item.direction}|"
        return [r for r in self.run.request_ids_by_doc.get(self.doc_id, []) if r.startswith(prefix)]

    def fail(self, exc) -> DocumentResult:
        reason = exc.reason if isinstance(exc, _DocFailure) else f"exception:{type(exc).__name__}"
        tb = None if isinstance(exc, _DocFailure) else traceback.format_exc(limit=8)
        self.run.log_error(self, {"kind": "document_failed", "error": reason, "detail": repr(exc), "traceback": tb})
        out_path = self.item.output_path
        if out_path is not None and Path(out_path).exists():
            Path(out_path).unlink()
        record = {
            "run_id": self.run.run_id, "direction": self.item.direction, "doc_id": self.doc_id,
            "sample_id": self.item.sample_id, "category": self.item.category, "status": "failed",
            "error": reason, "detail": repr(exc)[:500], "output_path": "", "source_path": self.item.rel_source,
            "source_chars": len(self.src), "roundtrip": dict(self.roundtrip), "nodes": len(self.nodes),
            "units": len(self.units), "requests": 0, "recovery_units": 0, "validator_available": self.validator_ok,
            "seconds": round(time.time() - self.t0, 3), "backend": self.run.backend_name,
        }
        self.run.writer.status(record)
        self.run.on_doc_done(self, record)
        self.result = DocumentResult(output_path="", status="failed", diagnostics=record)
        return self.result

    def roundtrip_record(self) -> dict:
        return {
            "sample_id": self.item.sample_id, "direction": self.item.direction, "source_path": self.item.rel_source,
            "source_sha256": sha256_text(self.src) if self.src else None, "chars": len(self.src),
            "roundtrip_ok": self.roundtrip.get("ok"), "offsets_ok": self.roundtrip.get("offsets_ok"),
            "nodes": len(self.nodes), "translatable": sum(1 for n in self.nodes if n.translatable),
            "copy_nodes": sum(1 for n in self.nodes if not n.translatable),
            "low_confidence_nodes": sum(1 for n in self.nodes if n.confidence < 0.5),
            "kinds": dict(Counter(n.kind for n in self.nodes)),
            "scopes": dict(Counter(n.scope_id for n in self.nodes)),
            "flags": dict(Counter(f for n in self.nodes for f in (n.flags or []))),
            "work_nodes": len(self.work), "units": len(self.units),
            "uncovered_nodes": len(self.uncovered),
        }

class _DocFailure(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason

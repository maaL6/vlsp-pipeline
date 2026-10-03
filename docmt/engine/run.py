"""Split from lam_docmt/runtime.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

import concurrent.futures as cf
import time
from collections import Counter, deque
from pathlib import Path

from ..core.codes import _stats
from ..core.datatypes import GenResult
from ..core.hashing import code_sha256, sha256_file, sha256_text
from .doc_job import _DocJob
from .scheduler import _Runner
from ..tracing.cache import ResumeCache
from ..tracing.writer import TEMPLATE, TIMEZONE

# -- one direction of a run ------------------------------------------------------------------------
class _Run:
    def __init__(self, cfg, tok, backend, comp, writer, cache, run_id, glossary=None, progress=None):
        self.cfg, self.tok, self.backend, self.comp = cfg, tok, backend, comp
        self.writer, self.cache, self.run_id = writer, cache, run_id
        self.glossary = glossary
        self.progress = progress
        self.backend_name = getattr(backend, "name", "unknown") if backend is not None else "none"
        self.cached_ids: set = set()
        self.op_retries: dict = {}
        self.request_ids_by_doc: dict = {}
        self.totals = Counter()
        self.jobs: list = []
        self.docs_done = 0
        self.t0 = time.time()
        self.model_sha = cfg.model_sha256 or ""
        self.decode = backend.decode_config() if backend is not None else {}

    def require(self, dry_run=False):
        """Resolve the collaborators a run needs before any request (fail fast)."""
        cfg = self.cfg
        names = ["parse_document", "split_long_nodes", "pack_units", "output_budget", "marker_namespace",
                 "recovery_units"]
        if not dry_run:
            names += ["parse_output", "check_unit", "check_document"]
        if cfg.source_context:
            names.append("SourceIndex")
        if cfg.source_context or cfg.soft_glossary or cfg.target_history:
            names += ["select_evidence", "render_background"]
        if cfg.target_history:
            names.append("Memory")
        for n in names:
            getattr(self.comp, n)

    def log_error(self, job, record):
        rec = {"run_id": self.run_id, "direction": job.item.direction if job else None,
               "doc_id": job.doc_id if job else None}
        rec.update(record)
        self.writer.error(rec)

    def cache_key(self, req) -> str:
        return ResumeCache.make_key(req.prompt, req.max_new_tokens, self.decode, self.model_sha)

    def lookup(self, req):
        if self.cache is None:
            return None
        hit = self.cache.get(self.cache_key(req), req.request_id)
        if hit is not None:
            self.cached_ids.add(req.request_id)
            self.totals["cache_hits"] += 1
            self._count_request(req, hit, cached=True)
        return hit

    def after(self, req, res):
        n = self.backend.pop_retries(req.request_id) if hasattr(self.backend, "pop_retries") else 0
        if n:
            self.op_retries[req.request_id] = n
        if self.cache is not None:
            self.cache.put(self.cache_key(req), res)
        self._count_request(req, res, cached=False)
        return res

    def _count_request(self, req, res, cached):
        doc = (req.meta or {}).get("doc_id")
        self.request_ids_by_doc.setdefault(doc, []).append(req.request_id)
        self.totals["requests"] += 1
        if (req.meta or {}).get("attempt", 0) >= 1:
            self.totals["recovery_requests"] += 1
        if not cached:
            self.totals["requests_sent"] += 1
            self.totals["prompt_tokens"] += int(res.input_tokens or 0)
            self.totals["generated_tokens"] += int(res.output_tokens or 0)
        if res.stop_type == "error":
            self.totals["op_error_results"] += 1

    def on_doc_done(self, job, record):
        self.docs_done += 1
        if self.progress:
            self.progress(f"[{job.item.direction}] {self.docs_done}/{len(self.jobs)} {job.doc_id}: "
                          f"{record.get('status')} requests={record.get('requests', 0)} "
                          f"t={time.time() - self.t0:.0f}s")

    def call(self, reqs):
        return self.backend.generate(reqs)

    def run_items(self, items) -> list:
        """Translate doc-direction items; returns DocumentResults in item order."""
        self.require()
        self.jobs = [_DocJob(self, it) for it in items]
        limit = 1 if self.cfg.target_history else None
        runners = [_Runner(j, limit) for j in self.jobs]
        self._schedule(runners)
        return [j.result for j in self.jobs]

    def _schedule(self, runners):
        bs = max(1, int(self.cfg.batch_size))
        batching = bool(getattr(self.backend, "batching", False))
        for r in runners:
            r.start()
        pool = cf.ThreadPoolExecutor(max_workers=1 if batching else bs, thread_name_prefix="gen")
        inflight = {}
        try:
            while True:
                if batching:
                    if not inflight:
                        batch = []
                        progressed = True
                        while progressed and len(batch) < bs:
                            progressed = False
                            for r in runners:
                                while len(batch) < bs and r.ready():
                                    req = r.take()
                                    hit = self.lookup(req)
                                    progressed = True
                                    if hit is not None:
                                        r.deliver(req, hit)
                                    else:
                                        batch.append((r, req))
                        if batch:
                            inflight[pool.submit(self.call, [q for _, q in batch])] = batch
                else:
                    progressed = True
                    while progressed:
                        progressed = False
                        for r in runners:
                            while sum(len(v) for v in inflight.values()) < bs and r.ready():
                                req = r.take()
                                hit = self.lookup(req)
                                progressed = True
                                if hit is not None:
                                    r.deliver(req, hit)
                                else:
                                    inflight[pool.submit(self.call, [req])] = [(r, req)]
                if not inflight:
                    if all(r.done for r in runners):
                        break
                    if any(r.ready() for r in runners):
                        continue
                    raise RuntimeError("scheduler stalled: documents wait for results that were never sent")
                done, _ = cf.wait(list(inflight), return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    pairs = inflight.pop(fut)
                    try:
                        results = list(fut.result())
                        if len(results) != len(pairs):
                            raise RuntimeError(f"backend returned {len(results)} results for {len(pairs)} requests")
                        by_id = {x.request_id: x for x in results}
                        if len(by_id) == len(pairs) and all(q.request_id in by_id for _, q in pairs):
                            results = [by_id[q.request_id] for _, q in pairs]
                    except Exception as exc:
                        self.writer.error({"run_id": self.run_id, "kind": "backend_exception", "error": repr(exc),
                                           "request_ids": [q.request_id for _, q in pairs]})
                        results = [GenResult(request_id=q.request_id, text="", eos=False, stop_type="error",
                                             input_tokens=0, output_tokens=0, error=f"backend exception: {exc!r}",
                                             backend=self.backend_name) for _, q in pairs]
                    for (r, q), res in zip(pairs, results):
                        r.deliver(q, self.after(q, res))
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown(wait=True)

    # ---- dry run -------------------------------------------------------------------------------
    def plan_items(self, items) -> dict:
        self.require(dry_run=True)
        stats = {"documents": 0, "failed": [], "nodes": 0, "units": 0, "unit_kinds": Counter(),
                 "budget_split": 0, "unfittable": 0, "roundtrip_failed": []}
        src_t, prm_t, new_t, ev_t = [], [], [], []
        example = None
        for it in items:
            job = _DocJob(self, it)
            try:
                job.prepare()
            except Exception as exc:
                stats["failed"].append({"doc_id": it.sample_id, "error": repr(exc)})
                if not job.roundtrip.get("ok"):
                    stats["roundtrip_failed"].append(it.sample_id)
                continue
            stats["documents"] += 1
            stats["nodes"] += len(job.nodes)
            stats["units"] += len(job.units)
            stats["budget_split"] += job.flags.get("budget_split", 0)
            for u in job.units:
                stats["unit_kinds"][u.kind] += 1
                b = job._build(u, 0, allow_history=False)
                rec = {"run_id": self.run_id, "direction": it.direction, "doc_id": it.sample_id,
                       "unit_id": u.unit_id, "unit_kind": u.kind, "scope_id": u.scope_id,
                       "node_ids": list(u.node_ids), "markers": len(u.markers), "source_tokens": u.source_tokens,
                       "split_from": u.split_from, "fits": b is not None}
                if b is None:
                    stats["unfittable"] += 1
                else:
                    rec.update(prompt_tokens=b.prompt_tokens, max_new_tokens=b.request.max_new_tokens,
                               total_tokens=b.prompt_tokens + b.request.max_new_tokens + self.cfg.margin,
                               prompt_sha256=sha256_text(b.request.prompt), prompt_style=b.template,
                               evidence=[{"kind": e.kind, "origin": e.origin, "tokens": e.tokens} for e in b.items],
                               evidence_dropped=[{"kind": e.kind, "tokens": e.tokens} for e in b.dropped])
                    prm_t.append(b.prompt_tokens)
                    new_t.append(b.request.max_new_tokens)
                    ev_t.append(sum(e.tokens for e in b.items))
                    if example is None:
                        example = b.request.prompt
                src_t.append(u.source_tokens)
                self.writer.plan(rec)
        stats["unit_kinds"] = dict(stats["unit_kinds"])
        stats.update(source_tokens=_stats(src_t), prompt_tokens=_stats(prm_t), max_new_tokens=_stats(new_t),
                     evidence_tokens=_stats(ev_t), example_prompt=example,
                     note="dry run: history evidence is empty (no accepted outputs yet)")
        return stats

    # ---- run.json ------------------------------------------------------------------------------
    def run_json(self, direction, items, results, utc_started, utc_finished, seconds, manifest_info,
                 stats_before, stats_after) -> dict:
        cfg = self.cfg
        desc = self.backend.describe() if self.backend is not None else {}
        statuses = Counter(r.status for r in results if r is not None)
        diag = [r.diagnostics for r in results if r is not None]
        delta = {k: stats_after.get(k, 0) - stats_before.get(k, 0) for k in set(stats_after) | set(stats_before)}
        tpl = Path(cfg.tokenizer_path).parent / "chat_template.jinja" if cfg.tokenizer_path else None
        return {
            "run_id": self.run_id, "branch": cfg.branch, "owner": cfg.owner,
            "utc_started": utc_started, "utc_finished": utc_finished, "timezone": TIMEZONE,
            "model": cfg.model, "model_sha256": cfg.model_sha256 or None,
            "model_provenance": (cfg.extra or {}).get("model_provenance", "not_checked"),
            "parameter_count": desc.get("parameter_count"),
            "parameter_count_source": desc.get("parameter_count_source", "not_checked"),
            "tokenizer_sha256": cfg.tokenizer_sha256 or (self.tok.sha256 if self.tok is not None else None),
            "template": TEMPLATE,
            "template_sha256": sha256_file(tpl) if tpl is not None and tpl.exists() else None,
            "prompt_style": cfg.prompt_style,
            "input_manifest": manifest_info.get("path"), "input_manifest_sha256": manifest_info.get("sha256"),
            "config": cfg.to_dict(), "config_sha256": cfg.sha256(), "code_sha256": code_sha256(),
            "direction": direction, "documents": len(items),
            "nodes": sum(d.get("nodes", 0) for d in diag), "units": sum(d.get("units", 0) for d in diag),
            "requests": self.totals["requests"], "requests_sent": self.totals["requests_sent"],
            "cache_hits": self.totals["cache_hits"], "recovery_requests": self.totals["recovery_requests"],
            "decode_config": self.decode, "device": desc.get("device"), "dtype": desc.get("dtype"),
            "runtime": round(seconds, 3), "prompt_tokens": self.totals["prompt_tokens"],
            "generated_tokens": self.totals["generated_tokens"], "peak_vram": desc.get("peak_vram"),
            "retry_count": sum(d.get("recovery_units", 0) for d in diag),
            "oom_count": delta.get("oom", 0), "op_error_count": delta.get("op_errors", 0),
            "op_retry_count": delta.get("op_retries", 0),
            "statuses": dict(statuses),
            "unresolved_units": sum(len(d.get("unresolved_units") or []) for d in diag),
            "unresolved_empty_nodes": sum(len(d.get("unresolved_empty") or []) for d in diag),
            "outputs": [d.get("output_path") for d in diag if d.get("output_path")],
            "backend": desc, "components": self.comp.origin(), "reference_used": False,
            "counters_note": ("retry_count = units that entered the single quality-recovery round; "
                              "recovery_requests = backend requests of that round; op_retry_count = network "
                              "re-sends; oom_count = CUDA OOM events (batch halved); op_error_count = requests "
                              "still failing after operational retries; prompt/generated tokens exclude cache hits"),
        }

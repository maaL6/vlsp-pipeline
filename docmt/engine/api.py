from __future__ import annotations

import json
import time
from pathlib import Path

from ..core.hashing import model_sha256, sha256_file, sha256_text, utc_now, write_json
from .doc_job import _as_components, _as_config
from .run import _Run
from ..inputs.sources import DIRECTIONS, DocItem, expand_rows, norm_directions, prediction_relpath
from ..tracing.cache import ResumeCache
from ..tracing.records import environment_text
from ..tracing.smoke_table import write_smoke_table
from ..tracing.writer import TEMPLATE, TraceWriter

# -- public API ----------------------------------------------------------------------------------------
def translate_document(input_path, output_path, src_lang, tgt_lang, config, backend=None, *, tok=None,
                       components=None, log_dir=None, doc_id=None, run_id=None) -> DocumentResult:
    """Translate one Markdown document (draft interface). Logs go to `log_dir` (or
    config.extra["log_dir"]) when given; otherwise diagnostics are only returned."""
    cfg = _as_config(config)
    if tok is None:
        from ..core.tok import HyTok
        tok = HyTok(cfg.tokenizer_path)
    if backend is None:
        from ..backends import make_backend
        backend = make_backend(cfg.backend, cfg, tok)
    log_dir = log_dir or (cfg.extra or {}).get("log_dir")
    writer = TraceWriter(log_dir)
    backend.error_sink = writer.error
    p = Path(input_path)
    sid = doc_id or f"{p.parent.name}/{p.stem}"
    item = DocItem(sample_id=sid, category=sid.split("/")[0], direction=f"{src_lang}-{tgt_lang}",
                   src_lang=src_lang, tgt_lang=tgt_lang, source_path=p, rel_source=str(p),
                   output_path=Path(output_path), rel_output=str(output_path))
    comp = _as_components(components)
    cache = ResumeCache(writer, cfg.sha256(), enabled=bool((cfg.extra or {}).get("resume", True)))
    glossary = _load_glossary(cfg, comp)
    run = _Run(cfg, tok, backend, comp, writer, cache, run_id or cfg.run_id, glossary=glossary)
    backend.prepare()
    return run.run_items([item])[0]

def _load_glossary(cfg, comp):
    if not cfg.soft_glossary:
        return None
    if not cfg.glossary_path:
        raise ValueError("soft_glossary=True needs glossary_path")
    return comp.Glossary.load(cfg.glossary_path)

def _manifest_info(rows, manifest_path) -> dict:
    if manifest_path:
        return {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
    blob = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    return {"path": None, "sha256": sha256_text(blob)}

def _fill_hashes(cfg, tok, backend_name):
    if tok is not None and not cfg.tokenizer_sha256:
        cfg.tokenizer_sha256 = tok.sha256
    if tok is not None and not cfg.tokenizer_path:
        cfg.tokenizer_path = str(tok.path)
    if cfg.model and not cfg.model_sha256 and backend_name != "mock" and Path(cfg.model).exists():
        cfg.model_sha256 = model_sha256(cfg.model)["sha256"]

README_STUB = """# Run {run_id}

Trạng thái: **{state}** (tự sinh bởi `python -m docmt run`; owner cập nhật sau khi kiểm).

- Branch: `{branch}` — backend: `{backend}` — owner: {owner}
- Thời điểm bắt đầu (UTC; team UTC+7): {utc}
- Manifest: `{manifest}` (sha256 `{manifest_sha}`)
- Hướng: {directions}
- Config: `run_config.json` (sha256 `{config_sha}`)

## Cấu trúc

- `predictions/<hướng>/<category>/<id>.md`: Markdown dịch (không marker nội bộ, không diagnostics).
- `logs/<hướng>.run.json`: metadata run; `logs/generation_trace.jsonl`: một bản ghi mỗi request;
  `logs/prompts.jsonl`: prompt theo sha256; `logs/document_status.jsonl`; `logs/runtime_errors.jsonl`;
  `logs/cache.jsonl`: cache resume.
- `tests/parser_roundtrip.json`, `tests/smoke_result.csv`.

## Lưu ý

- Chỉ đọc nguồn khi sinh; reference không được đọc. Cảnh báo tự động là `needs_review`, không phải
  kết luận sai nghĩa. Chưa có review nghĩa thì ghi `not_checked`.
- Metric/thời gian/VRAM chỉ ghi khi đã đo (xem run.json).
"""

CHANGE_NOTES_STUB = """# Change notes — {run_id}

Trạng thái: **not_checked** (stub tự sinh; owner điền).

- So với run trước / nhánh gốc: (chưa điền)
- Thay đổi config (một thành phần mỗi ablation): (chưa điền)
- Đã sửa / chưa sửa: (chưa điền)
"""

def _write_stubs(out_dir: Path, cfg, manifest_info, directions, backend_name):
    write_json(out_dir / "run_config.json", {**cfg.to_dict(), "config_sha256": cfg.sha256()})
    readme = out_dir / "README.md"
    if not readme.exists():
        readme.write_text(README_STUB.format(
            run_id=cfg.run_id, state="running", branch=cfg.branch, backend=backend_name, owner=cfg.owner,
            utc=utc_now(), manifest=manifest_info.get("path"), manifest_sha=manifest_info.get("sha256"),
            directions=", ".join(directions), config_sha=cfg.sha256()), encoding="utf-8")
    notes = out_dir / "change_notes.md"
    if not notes.exists():
        notes.write_text(CHANGE_NOTES_STUB.format(run_id=cfg.run_id), encoding="utf-8")

def _write_model_manifest(out_dir: Path, cfg, desc: dict, tok):
    write_json(out_dir / "model_manifest.json", {
        "status": "completed" if desc.get("parameter_count") is not None else "not_checked",
        "backend": desc.get("backend"), "model": cfg.model, "model_sha256": cfg.model_sha256 or None,
        "model_provenance": (cfg.extra or {}).get("model_provenance", "not_checked"),
        "parameter_count": desc.get("parameter_count"),
        "parameter_count_source": desc.get("parameter_count_source", "not_checked"),
        "tokenizer_path": cfg.tokenizer_path, "tokenizer_sha256": cfg.tokenizer_sha256 or None,
        "template": TEMPLATE, "eos": "<｜hy_place▁holder▁no▁2｜> (id 120020)",
        "adapter": (cfg.extra or {}).get("adapter"), "backend_describe": desc, "utc": utc_now(),
    })

def _check_run_dir(out_dir: Path, cfg):
    p = out_dir / "run_config.json"
    if p.exists():
        try:
            old = json.loads(p.read_text(encoding="utf-8")).get("config_sha256")
        except Exception:
            old = None
        if old and old != cfg.sha256() and not (cfg.extra or {}).get("allow_config_change"):
            raise ValueError(f"{out_dir} already holds a run with config sha {old[:12]}...; the contract "
                             "requires a new run_id for a different config (or set extra.allow_config_change)")

def run_manifest(rows, data_root, out_dir, cfg, backend, tok, directions=DIRECTIONS, *, components=None,
                 manifest_path=None, limit=None, dry_run=False, progress=None) -> dict:
    """Run every doc-direction of `rows` and write the delivery layout under `out_dir`
    (= delivery/<run_id>). Returns {direction: summary} (dry run: {direction: plan stats})."""
    out_dir = Path(out_dir)
    dirs = norm_directions(directions)
    comp = _as_components(components)
    backend_name = getattr(backend, "name", cfg.backend) if backend is not None else cfg.backend
    _fill_hashes(cfg, tok, "mock" if dry_run else backend_name)
    manifest_info = _manifest_info(rows, manifest_path)
    items_by_dir = {}
    for d in dirs:
        its = expand_rows(rows, data_root, [d])
        if limit:
            its = its[: int(limit)]
        for it in its:
            it.rel_output = prediction_relpath(d, it.sample_id)
            it.output_path = out_dir / it.rel_output
        items_by_dir[d] = its
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    writer = TraceWriter(out_dir / "logs")
    if dry_run:
        writer.reset(dirs)
        plan = {}
        for d in dirs:
            run = _Run(cfg, tok, None, comp, writer, None, cfg.run_id)
            plan[d] = run.plan_items(items_by_dir[d])
        write_json(out_dir / "logs" / "plan_stats.json", {"run_id": cfg.run_id, "utc": utc_now(),
                                                          "config_sha256": cfg.sha256(), "directions": plan})
        return plan
    _check_run_dir(out_dir, cfg)
    (out_dir / "predictions").mkdir(exist_ok=True)
    (out_dir / "tests").mkdir(exist_ok=True)
    writer.reset(dirs)
    writer.touch_all()
    _write_stubs(out_dir, cfg, manifest_info, dirs, backend_name)
    (out_dir / "environment.txt").write_text(environment_text(None, {"run_id": cfg.run_id}), encoding="utf-8")
    backend.error_sink = writer.error
    backend.prepare()
    cache = ResumeCache(writer, cfg.sha256(), enabled=bool((cfg.extra or {}).get("resume", True)))
    glossary = _load_glossary(cfg, comp)
    summary, roundtrip = {}, []
    for d in dirs:
        items = items_by_dir[d]
        if not items:
            continue
        run = _Run(cfg, tok, backend, comp, writer, cache, cfg.run_id, glossary=glossary, progress=progress)
        before = backend.stats_snapshot() if hasattr(backend, "stats_snapshot") else {}
        started, t0 = utc_now(), time.time()
        results = run.run_items(items)
        after = backend.stats_snapshot() if hasattr(backend, "stats_snapshot") else {}
        info = run.run_json(d, items, results, started, utc_now(), time.time() - t0, manifest_info, before, after)
        write_json(out_dir / "logs" / f"{d}.run.json", info)
        roundtrip += [j.roundtrip_record() for j in run.jobs]
        summary[d] = {k: info[k] for k in ("documents", "requests", "recovery_requests", "statuses",
                                           "unresolved_units", "op_error_count", "runtime")}
    _write_roundtrip(out_dir / "tests" / "parser_roundtrip.json", roundtrip, dirs, cfg.run_id)
    desc = backend.describe()
    (out_dir / "environment.txt").write_text(environment_text(desc, {"run_id": cfg.run_id}), encoding="utf-8")
    _write_model_manifest(out_dir, cfg, desc, tok)
    all_items = [{"sample_id": it.sample_id, "direction": it.direction} for d in dirs for it in items_by_dir[d]]
    write_smoke_table(out_dir, all_items, run_id=cfg.run_id)
    return summary

def _write_roundtrip(path: Path, records, directions, run_id):
    old = []
    if path.exists():
        try:
            old = [r for r in json.loads(path.read_text(encoding="utf-8")).get("documents", [])
                   if r.get("direction") not in set(directions)]
        except Exception:
            old = []
    docs = old + records
    ok = all(r.get("roundtrip_ok") and r.get("offsets_ok") for r in docs) if docs else False
    write_json(path, {"run_id": run_id, "utc": utc_now(), "status": "completed" if docs else "not_checked",
                      "all_roundtrip_ok": ok, "documents": docs,
                      "note": "exact raw roundtrip of parse nodes (''.join(n.raw) == source) and contiguous offsets"})

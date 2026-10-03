"""Logs of a run (SPEC section 3, contract section 2/3): trace/status/error/prompt/cache JSONL
writers, hashes, environment text, resume cache and the smoke-result table.

Nothing here reads reference/gold text: the smoke table is built from document_status.jsonl and
generation_trace.jsonl only.
"""
# Split from lam_docmt/trace.py (statements verbatim; tools/split_from_lam_docmt.py).
from __future__ import annotations

import json
import threading
from pathlib import Path

from ..core.hashing import read_jsonl, utc_now

# One record per backend request (SPEC section 3 "Trace record").
TRACE_KEYS = (
    "run_id", "direction", "doc_id", "scope_id", "unit_id", "unit_kind", "attempt", "parent_unit_id",
    "node_ids", "source_offsets", "raw_source_sha256", "normalized_source", "evidence",
    "prompt_sha256", "prompt_style", "source_tokens", "input_tokens", "output_tokens",
    "max_new_tokens", "eos", "stop_type", "output", "segments", "hard_failures", "warnings",
    "retry_reason", "selected", "seconds", "backend",
)

# <direction>.run.json (SPEC section 3 "run.json keys").
RUN_KEYS = (
    "run_id", "branch", "owner", "utc_started", "utc_finished", "timezone", "model", "model_sha256",
    "model_provenance", "parameter_count", "tokenizer_sha256", "template", "prompt_style",
    "input_manifest", "input_manifest_sha256", "config", "config_sha256", "code_sha256", "direction",
    "documents", "nodes", "units", "requests", "recovery_requests", "decode_config", "device", "dtype",
    "runtime", "prompt_tokens", "generated_tokens", "peak_vram", "retry_count", "oom_count",
    "op_error_count",
)

STATUS_VALUES = ("proposed", "not_started", "running", "completed", "failed", "not_checked")

SMOKE_COLUMNS = ("sample_id", "direction", "run_id", "output_path", "source_span", "coverage_status",
                 "marker_eos_status", "table_number_status", "name_status", "hard_failures", "warnings",
                 "retry_count", "evidence", "review_required", "notes")

CHECK_VALUES = ("pass", "fail", "needs_review", "not_checked")

TEMPLATE = "hy_a1 chat_template.jinja, add_generation_prompt"

TIMEZONE = "UTC; team UTC+7"

LOG_FILES = {
    "trace": "generation_trace.jsonl",
    "status": "document_status.jsonl",
    "errors": "runtime_errors.jsonl",
    "prompts": "prompts.jsonl",
    "cache": "cache.jsonl",
    "plan": "plan.jsonl",
}

_PER_DIRECTION = ("trace", "status", "errors", "plan")

# -- writers ---------------------------------------------------------------------------------------
class TraceWriter:
    """Thread-safe JSONL writers for logs/ (append + flush per record).

    `logs_dir=None` keeps everything in memory (`self.records[kind]`), used by
    translate_document() without a log directory and by tests."""

    def __init__(self, logs_dir=None):
        self.logs_dir = Path(logs_dir) if logs_dir else None
        self._lock = threading.Lock()
        self.records = {k: [] for k in LOG_FILES}
        self._seen_prompts = set()
        if self.logs_dir is not None:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            for rec in read_jsonl(self.path("prompts")):
                self._seen_prompts.add(rec.get("prompt_sha256"))

    def path(self, kind: str):
        return None if self.logs_dir is None else self.logs_dir / LOG_FILES[kind]

    def touch_all(self):
        if self.logs_dir is not None:
            for kind in LOG_FILES:
                if kind != "plan":
                    self.path(kind).touch(exist_ok=True)

    def reset(self, directions):
        """Drop records of `directions` from per-direction logs (a re-run replaces them);
        prompts.jsonl and cache.jsonl are kept (resume)."""
        directions = set(directions)
        with self._lock:
            for kind in _PER_DIRECTION:
                self.records[kind] = [r for r in self.records[kind] if r.get("direction") not in directions]
                p = self.path(kind)
                if p is None or not p.exists():
                    continue
                keep = [r for r in read_jsonl(p) if r.get("direction") not in directions]
                p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")

    def _write(self, kind: str, record: dict):
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            self.records[kind].append(record)
            p = self.path(kind)
            if p is not None:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(line)

    def trace(self, record: dict):
        missing = [k for k in TRACE_KEYS if k not in record]
        if missing:
            raise KeyError(f"trace record misses keys {missing}")
        self._write("trace", record)

    def prompt(self, prompt_sha: str, prompt: str):
        with self._lock:
            if prompt_sha in self._seen_prompts:
                return
            self._seen_prompts.add(prompt_sha)
        self._write("prompts", {"prompt_sha256": prompt_sha, "prompt": prompt})

    def status(self, record: dict):
        self._write("status", record)

    def error(self, record: dict):
        rec = {"utc": utc_now()}
        rec.update(record)
        self._write("errors", rec)

    def plan(self, record: dict):
        self._write("plan", record)

    def cache_line(self, record: dict):
        self._write("cache", record)

"""Generation backends (SPEC section 3): GenRequest list -> GenResult list.

- hf:       transformers AutoModelForCausalLM, padded batch generate (torch imported lazily).
- mock:     deterministic scripted fake for tests (echo + injectable faults).

Operational problems (network, OOM) are retried/counted here and never mixed with the
quality-recovery round of the runtime: `stats` = {"op_retries", "op_errors", "oom"}.
"""
# Split from lam_docmt/backends/__init__.py (statements verbatim; tools/split_from_lam_docmt.py).
from __future__ import annotations

import threading
import time

from ..core.datatypes import GenResult

BACKENDS = ("hf", "mock")

class Backend:
    """Base class.

    `batching` tells the runtime scheduler how to dispatch:
      True  -> one generate() call at a time with up to batch_size requests (padded HF batch);
      False -> single-request generate() calls, up to batch_size of them in parallel threads
               (llama.cpp server slots, mock).
    """

    name = "base"
    batching = False

    def __init__(self):
        self._lock = threading.Lock()
        self.stats = {"op_retries": 0, "op_errors": 0, "oom": 0}
        self.retries_by_request: dict = {}
        self.error_sink: Optional[Callable[[dict], None]] = None

    # -- interface -------------------------------------------------------------------------
    def generate(self, requests):  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> dict:
        return {"backend": self.name}

    def decode_config(self) -> dict:
        """Decoding parameters that change outputs (part of the resume-cache key)."""
        return {"backend": self.name, "do_sample": False}

    def prepare(self):
        """Load weights / check the server before the first request (no-op by default)."""

    def close(self):
        pass

    # -- helpers ---------------------------------------------------------------------------
    def bump(self, key: str, n: int = 1):
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + n

    def note_retry(self, request_id: str):
        with self._lock:
            self.retries_by_request[request_id] = self.retries_by_request.get(request_id, 0) + 1
            self.stats["op_retries"] = self.stats.get("op_retries", 0) + 1

    def pop_retries(self, request_id: str) -> int:
        with self._lock:
            return self.retries_by_request.pop(request_id, 0)

    def stats_snapshot(self) -> dict:
        with self._lock:
            return dict(self.stats)

    def log_error(self, record: dict):
        rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "backend": self.name}
        rec.update(record)
        sink = self.error_sink
        if sink is not None:
            try:
                sink(rec)
            except Exception:  # logging must never break generation
                pass

    def error_result(self, req, error: str, seconds: float = 0.0) -> GenResult:
        return GenResult(request_id=req.request_id, text="", eos=False, stop_type="error",
                         input_tokens=0, output_tokens=0, seconds=round(seconds, 3),
                         error=error, backend=self.name)

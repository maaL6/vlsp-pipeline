from __future__ import annotations

import json
import threading
from dataclasses import asdict

from ..core.datatypes import GenResult
from ..core.hashing import read_jsonl, sha256_text, utc_now

class ResumeCache:
    """logs/cache.jsonl: key = sha256(prompt, max_new_tokens, decode cfg, model sha). Entries are
    reused only when their run-config sha equals the current one. Error results are never cached."""

    def __init__(self, writer: TraceWriter | None, config_sha: str, enabled: bool = True):
        self.writer = writer
        self.config_sha = config_sha
        self.enabled = enabled
        self.entries: dict = {}
        self.hits = 0
        self._lock = threading.Lock()
        if enabled and writer is not None and writer.path("cache") is not None:
            for rec in read_jsonl(writer.path("cache")):
                if rec.get("config_sha256") == config_sha and "key" in rec and "result" in rec:
                    self.entries[rec["key"]] = rec["result"]
        self.loaded = len(self.entries)

    @staticmethod
    def make_key(prompt: str, max_new_tokens: int, decode: dict, model_sha: str) -> str:
        blob = json.dumps([prompt, int(max_new_tokens), decode, model_sha], ensure_ascii=False, sort_keys=True)
        return sha256_text(blob)

    def get(self, key: str, request_id: str):
        if not self.enabled:
            return None
        with self._lock:
            d = self.entries.get(key)
            if d is None:
                return None
            self.hits += 1
        d = dict(d)
        d["request_id"] = request_id
        return GenResult(**d)

    def put(self, key: str, result: GenResult):
        if not self.enabled or result.stop_type == "error" or result.error:
            return
        d = asdict(result)
        with self._lock:
            if key in self.entries:
                return
            self.entries[key] = d
        if self.writer is not None:
            self.writer.cache_line({"key": key, "config_sha256": self.config_sha, "utc": utc_now(), "result": d})

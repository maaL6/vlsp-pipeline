from __future__ import annotations

import threading
from collections import Counter

from .common import _strip_markers

class Memory:
    """Accepted target history keyed by (doc_id, scope_id). Only units accepted by the validator
    (zero hard failures) are stored; unresolved units are refused. It never crosses documents or
    scopes and never holds reference/gold text (callers pass model outputs only). doc_id should be
    the doc-direction key when both directions share one Memory."""

    def __init__(self):
        self._store = {}
        self._lock = threading.Lock()
        self._seq = 0
        self.refused = Counter()

    def add(self, doc_id, scope_id, unit_id, src_payload, tgt_text, accepted: bool, *,
            raw_span=None, meta=None) -> bool:
        if accepted is not True:
            self.refused["not_accepted"] += 1
            return False
        tgt_segments = _strip_markers(tgt_text)
        if not tgt_segments:
            self.refused["empty"] += 1
            return False
        src_segments = _strip_markers(src_payload)
        with self._lock:
            self._seq += 1
            entry = {"doc_id": doc_id, "scope_id": scope_id, "unit_id": unit_id,
                     "src": "\n".join(src_segments), "tgt": "\n".join(tgt_segments),
                     "src_segments": src_segments, "tgt_segments": tgt_segments,
                     "origin": "accepted_history", "status": "accepted",
                     "raw_span": list(raw_span) if raw_span else None,
                     "meta": dict(meta or {}), "seq": self._seq}
            self._store.setdefault((doc_id, scope_id), []).append(entry)
        return True

    def recent(self, doc_id, scope_id, n) -> list:
        """Up to n most recent accepted entries of this doc + scope, oldest first (copies)."""
        if not n or n <= 0:
            return []
        with self._lock:
            lst = list(self._store.get((doc_id, scope_id), ()))
        return [dict(e, src_segments=list(e["src_segments"]), tgt_segments=list(e["tgt_segments"]))
                for e in lst[-int(n):]]

    def forget(self, doc_id):
        with self._lock:
            for k in [k for k in self._store if k[0] == doc_id]:
                del self._store[k]

    def stats(self) -> dict:
        with self._lock:
            return {"entries": sum(len(v) for v in self._store.values()),
                    "keys": len(self._store), "refused": dict(self.refused)}

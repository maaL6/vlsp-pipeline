from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .common import _fold, _overlaps, _term_regex
from ..core.text_normalize import normalize

class Glossary:
    """Soft term hints from release_v2/glossary_provenance.jsonl (fields en, vi, domain, ...).
    Matching: case-insensitive, word-bounded, on the unit's source payload; longest first; one
    entry per source term; source terms shorter than 3 characters are skipped."""

    def __init__(self, entries, path=None, sha256=None):
        self.entries = list(entries)
        self.path = path
        self.sha256 = sha256
        self._tables = {lang: self._make_table(lang) for lang in ("en", "vi")}

    @classmethod
    def load(cls, path) -> "Glossary":
        p = Path(path)
        data = p.read_bytes()
        entries = []
        for lineno, line in enumerate(data.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            en = normalize(" ".join(str(r.get("en", "")).split()))
            vi = normalize(" ".join(str(r.get("vi", "")).split()))
            if not en or not vi:
                continue
            entries.append({"en": en, "vi": vi, "domain": r.get("domain"), "status": r.get("status"),
                            "audit_status": r.get("audit_status"),
                            "historical_origin": r.get("historical_origin"), "line": lineno})
        return cls(entries, path=str(p), sha256=hashlib.sha256(data).hexdigest())

    def _make_table(self, src_lang):
        tgt_lang = "vi" if src_lang == "en" else "en"
        seen, table = set(), []
        for e in self.entries:
            s, t = e[src_lang], e[tgt_lang]
            if len(s) < 3:
                continue
            key = _fold(s)
            if key in seen:
                continue
            seen.add(key)
            table.append((s, t, key, _term_regex(s), e))
        table.sort(key=lambda x: (-len(x[0]), x[2]))
        return table

    def match_entries(self, text, src_lang, max_terms) -> list:
        """[(src_term, tgt_term, entry)] longest first, non-overlapping, <= max_terms."""
        if not text or not max_terms or max_terms <= 0:
            return []
        lang = "vi" if str(src_lang).lower().startswith("vi") else "en"
        tf = _fold(text)
        consumed, out = [], []
        for s, t, key, rx, e in self._tables[lang]:
            if key not in tf:
                continue
            free = [(m.start(), m.end()) for m in rx.finditer(text)]
            free = [sp for sp in free if not _overlaps([sp], consumed)]
            if not free:
                continue
            consumed.extend(free)
            out.append((s, t, e))
            if len(out) >= max_terms:
                break
        return out

    def match(self, text, src_lang, max_terms) -> list:
        return [(s, t) for s, t, _e in self.match_entries(text, src_lang, max_terms)]

    def describe(self) -> dict:
        return {"path": self.path, "sha256": self.sha256, "entries": len(self.entries),
                "domains": dict(Counter(e["domain"] for e in self.entries).most_common()),
                "audit_status": dict(Counter(e["audit_status"] for e in self.entries)),
                "historical_origin": dict(Counter(e["historical_origin"] for e in self.entries)),
                "usable_terms": {k: len(v) for k, v in self._tables.items()},
                "use": "soft hints only (cfg.soft_glossary); provenance not verified"}

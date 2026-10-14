"""Hy-MT2 tokenizer helper: exact token counts without loading weights (tokenizers>=0.21)."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

class HyTok:
    def __init__(self, path):
        from tokenizers import Tokenizer

        self.path = Path(path)
        self.tk = Tokenizer.from_file(str(self.path))
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()

    @lru_cache(maxsize=65536)
    def count(self, text: str) -> int:
        """Tokens of plain text (no special tokens added). Added special tokens written in the
        text (e.g. <｜hy_User｜>) are matched as single tokens, as in the rendered chat prompt."""
        return len(self.tk.encode(text, add_special_tokens=False).ids)

    def encode(self, text: str):
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=True) -> str:
        return self.tk.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def truncate_tokens(self, text: str, limit: int) -> str:
        """Longest prefix of `text` with <= limit tokens (character binary search, no partial UTF-8)."""
        if self.count(text) <= limit:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.count(text[:mid]) <= limit:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]

"""Split from lam_docmt/validate.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

import re
import unicodedata

FFFD = "�"

ANY_MARKER = re.compile(r"\[SEG[0-9A-F]{0,4}_\d{6}\]")       # any marker shape (ours or source-native)

_LETTER = re.compile(r"[^\W\d_]")

_ALNUM = re.compile(r"[^\W_]")

def _nfc(text):
    # detection copy only (raw text is never modified in place)
    return unicodedata.normalize("NFC", text or "").replace("Ð", "Đ").replace("ð", "đ")

def _ws(text):
    return " ".join(text.split())

def _dedupe(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

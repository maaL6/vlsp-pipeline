from __future__ import annotations

import re

from ..core.text_normalize import normalize

def unwrap(text: str) -> str:
    """Unwrap: join soft-wrapped lines with one space; a line ending in letter+'-' joins without
    space (hyphen kept); collapse internal runs of spaces to one."""
    out = ""
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if not out:
            out = line
        elif len(out) >= 2 and out[-1] == "-" and out[-2].isalpha():
            out += line
        else:
            out += " " + line
    return re.sub(r"[ \t]+", " ", out)

def make_payload(content_raw: str) -> str:
    return normalize(unwrap(content_raw))

from __future__ import annotations

import collections
import difflib
import re

from .common import _nfc
from .numbers import _roman_value

# ============================================================================ headings
_KW_TYPE = {"article": "article", "điều": "article", "chapter": "chapter", "chương": "chapter",
            "section": "section", "mục": "section", "part": "part", "phần": "part"}

_HEADING_TYPES = ("part", "chapter", "section", "article")

_VI_NAME = {"article": "Điều", "chapter": "Chương", "section": "Mục", "part": "Phần"}

_KW_ALT = "article|điều|chapter|chương|section|mục|part|phần"

_HNUM = r"\d{1,4}(?:dd|[a-zđ])?(?!\d)|[IVXLC][IVXLCl]*|[A-Z]"

# heading line: keyword + number at line start, then ".", ".-", ":", "-", "–", "~", ")" or end of line
_HEAD_LINE = re.compile(r"^[ \t]*(?:[#>*_“\"'‘]+[ \t]*)*(" + _KW_ALT + r")[ \t]+(" + _HNUM + r")[ \t]*"
                        r"(?=\.-|[.:\-–~)]|$)", re.I | re.M)

# any mention (headings and in-text references such as "khoản 1 Điều 5")
_HEAD_ANY = re.compile(r"(?<![^\W\d_])(" + _KW_ALT + r")[ \t]+(" + _HNUM + r")(?![^\W_])", re.I)

# heading-like mention inside a line (flattened outputs): "... Điều 25. Hội nghị ..."
_HEAD_INLINE = re.compile(r"(?<![^\W\d_])(" + _KW_ALT + r")[ \t]+(" + _HNUM + r")[ \t]*(?:\.-|[.:])[ \t]*(\S)",
                          re.I)

def _hvalue(tok):
    t = (tok or "").strip()
    if not t:
        return None
    if t[0].isdigit():
        m = re.match(r"(\d+)(.*)", t)
        return str(int(m.group(1))) + m.group(2).lower().replace("dd", "đ")
    r = _roman_value(t)
    if r:
        return str(r)
    if len(t) == 1 and t.isalpha() and t.isascii():
        return t.upper()
    return None

def heading_list(text, mode="line"):
    """[(type, value)] in document order. mode: 'line' (line-start headings), 'any' (every
    mention), 'inline' (heading-like 'Điều N. Title' anywhere in a line)."""
    text = _nfc(text)
    out = []
    if mode == "inline":
        for m in _HEAD_INLINE.finditer(text):
            if not m.group(3).isupper():
                continue
            v = _hvalue(m.group(2))
            if v:
                out.append((_KW_TYPE[m.group(1).lower()], v))
        return out
    rx = _HEAD_LINE if mode == "line" else _HEAD_ANY
    for m in rx.finditer(text):
        v = _hvalue(m.group(2))
        if v:
            out.append((_KW_TYPE[m.group(1).lower()], v))
    return out

def _hsort(k):
    t, v = k
    m = re.match(r"(\d+)(.*)", v)
    return (_HEADING_TYPES.index(t), 0 if m else 1, int(m.group(1)) if m else 0, v)

def display_heading(k):
    return f"{_VI_NAME[k[0]]} {k[1]}"

def _hstr(k):
    return f"{k[0]}:{k[1]}"

def compare_headings(src, out):
    """Heading numbers (Article/Điều, Chapter/Chương, Section/Mục, Part/Phần; roman numerals
    compared by value) of the source vs the output.

    missing         : source heading number found nowhere in the output (not even as a reference)
    not_as_heading  : present in the output only as an in-text mention (format/flattening issue)
    fewer           : the source has the heading k times, the output fewer times as a heading
    duplicate       : the output has the heading more often than the source
    extra           : heading in the output whose number the source never mentions
    """
    s_line, o_line = heading_list(src, "line"), heading_list(out, "line")
    s_any = collections.Counter(heading_list(src, "any"))
    o_any = collections.Counter(heading_list(out, "any"))
    o_inline = heading_list(out, "inline")
    sc, oc = collections.Counter(s_line), collections.Counter(o_line)
    missing = sorted((k for k in sc if o_any[k] == 0), key=_hsort)
    not_as_heading = sorted((k for k in sc if o_any[k] > 0 and oc[k] == 0), key=_hsort)
    fewer = [{"key": _hstr(k), "src": sc[k], "out": oc[k]} for k in sorted(sc, key=_hsort) if 0 < oc[k] < sc[k]]
    duplicate = [{"key": _hstr(k), "src": sc[k], "out": oc[k]} for k in sorted(oc, key=_hsort)
                 if k in sc and oc[k] > sc[k]]
    cand = set(oc) | set(o_inline)
    extra = sorted((k for k in cand if k not in sc and s_any[k] == 0), key=_hsort)
    extra_mentioned = sorted((k for k in cand if k not in sc and s_any[k] > 0), key=_hsort)
    sm = difflib.SequenceMatcher(None, [_hstr(k) for k in s_line], [_hstr(k) for k in o_line], autojunk=False)
    ops = [{"op": tag, "src": [_hstr(k) for k in s_line[i1:i2]], "out": [_hstr(k) for k in o_line[j1:j2]]}
           for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
    return {"src_headings": len(s_line), "out_headings": len(o_line),
            "missing": [_hstr(k) for k in missing],
            "not_as_heading": [_hstr(k) for k in not_as_heading],
            "fewer": fewer, "duplicate": duplicate,
            "extra": [_hstr(k) for k in extra],
            "extra_mentioned_in_source": [_hstr(k) for k in extra_mentioned],
            "sequence_equal": not ops, "sequence_ops": ops[:20], "sequence_ops_total": len(ops)}

def _has_heading(text, htype, value):
    return any(t == htype and v == value for t, v in heading_list(text, "any"))

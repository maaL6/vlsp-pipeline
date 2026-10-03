"""Split from lam_docmt/validate.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

import collections
import re

from .common import _ALNUM, _nfc, _ws

LOOP_MIN_UNIT, LOOP_MAX_UNIT, LOOP_MIN_REPS = 4, 80, 5   # substring loop: 4-80 chars x >= 5

LINE_MIN_CHARS, LINE_MIN_COUNT = 8, 4                     # line loop: >= 8 chars, >= 4 times

PUNCT_LOOP_MIN_CHARS = 300    # separator-like runs (----, - - -, ....) only count when this long

# ============================================================================ loops
_LOOP_PATTERNS = {}

def _zero_run_re(n_bytes):
    rx = _LOOP_PATTERNS.get(n_bytes)
    if rx is None:
        rx = _LOOP_PATTERNS[n_bytes] = re.compile(b"\x00{%d,}" % n_bytes)
    return rx

def _periodic_regions(text, lo=LOOP_MIN_UNIT, hi=LOOP_MAX_UNIT, reps=LOOP_MIN_REPS):
    """Regions where a unit of lo..hi chars repeats >= reps times back to back (same result as
    the legacy regex (.{4,80}?)\\1{4,}, smallest unit first), found with a big-int XOR of the
    UTF-32 text against itself shifted by L characters (stdlib only, linear per L)."""
    n = len(text)
    found = []
    if n < lo * reps:
        return found
    data = text.encode("utf-32-le")
    big = int.from_bytes(data, "little")
    covered = []
    for L in range(lo, min(hi, n // reps) + 1):
        xb = (big ^ (big >> (32 * L))).to_bytes(len(data), "little")
        need = (reps - 1) * L                     # chars i with text[i] == text[i + L]
        for m in _zero_run_re(4 * need).finditer(xb):
            s, e = (m.start() + 3) // 4, min(m.end() // 4, n - L)
            if e - s < need:
                continue
            start, end = s, e + L
            if any(cs <= start and end <= ce for cs, ce in covered):
                continue                          # same region with a smaller period
            covered.append((start, end))
            found.append((start, end, L))
    found.sort()
    return found

def _loop_canon(unit):
    u = re.sub(r"\s+", " ", unit)
    if len(u) > 1 and u.startswith(" ") and u.endswith(" "):
        u = u[1:]
    return min(u[i:] + u[:i] for i in range(len(u))).casefold() if u else u

def _loop_regions(text):
    out = []
    line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
    import bisect
    for start, end, L in _periodic_regions(text):
        unit = text[start:start + L]
        cls = "text" if _ALNUM.search(unit) else "punct"
        out.append({"start": start, "line": bisect.bisect_right(line_starts, start), "unit": unit,
                    "unit_len": L, "reps": (end - start) // L, "chars": end - start, "class": cls,
                    "canon": _loop_canon(unit)})
    return out

_LINE_LABEL = re.compile(r"^[“\"'‘(\[]*\s*(?:\(\s*(\d{1,3}|[a-zđ]|dd|[ivxlc]{1,6})\s*\)|"
                         r"(\d{1,3}(?:\.\d{1,2})*|[a-zđ]|dd|[ivxlc]{1,6})\s*[.)\-–:])(?=\s|$)", re.I)

_COLUMNAR = re.compile(r"\S(?: {2,}|\t)\S")

def _label_core(line):
    m = _LINE_LABEL.match(line)
    if not m:
        return ""
    return (m.group(1) or m.group(2)).lower().replace("dd", "đ")

def _source_units(text):
    """Source units for repeated-line comparison: unwrapped paragraphs (blank-line separated);
    lines of column-aligned blocks (tables) are kept as separate units."""
    units = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        if len(lines) > 1 and any(_COLUMNAR.search(ln.strip()) for ln in lines):
            units.extend(_ws(ln) for ln in lines)
        else:
            units.append(_ws(" ".join(lines)))
    return [u for u in units if len(u) >= LINE_MIN_CHARS and _ALNUM.search(u)]

def find_loops(src, out):
    """Repetition in the output that the source does not have (warning `loop_not_in_source`).

    substring: a 4-80 char unit repeated >= 5 times back to back. Units with letters/digits
      count unless the same unit also loops in the source (or the source has a letter/digit
      loop with at least as many repetitions); separator-like units (----, - - -, ....) count
      only when the run is >= 300 chars and the source has no such long run.
    line: an output line (>= 8 chars, ws-collapsed) occurring >= 4 times, more often than any
      source unit that could explain it: the same text, or (cross-lingual) the most repeated
      source unit with the same clause/point label ("a)", "1.", none). Legitimate repeated table
      rows and points such as "a) Issue warning;" x13 are therefore not loops.
    """
    src, out = _nfc(src), _nfc(out)
    s_reg, o_reg = _loop_regions(src), _loop_regions(out)
    s_text = [r for r in s_reg if r["class"] == "text"]
    s_canon = {r["canon"] for r in s_text}
    s_maxreps = max((r["reps"] for r in s_text), default=0)
    s_long_punct = any(r["chars"] >= PUNCT_LOOP_MIN_CHARS for r in s_reg if r["class"] == "punct")
    subs = []
    for r in o_reg:
        if r["class"] == "punct":
            if r["chars"] < PUNCT_LOOP_MIN_CHARS:
                continue                       # separators / layout spaces: normal
            flagged = not s_long_punct
            reason = "long separator run, none in source" if flagged else "source has long separator runs"
        elif r["canon"] in s_canon:
            flagged, reason = False, "same unit repeats in source"
        elif s_maxreps >= r["reps"]:
            flagged, reason = False, "source has a comparable letter/digit repetition"
        else:
            flagged, reason = True, "not in source"
        subs.append({"kind": "substring", "line": r["line"], "unit": _ws(r["unit"])[:80] or repr(r["unit"]),
                     "unit_len": r["unit_len"], "reps": r["reps"], "chars": r["chars"], "class": r["class"],
                     "flagged": flagged, "reason": reason})
    # repeated lines
    o_lines = [_ws(ln) for ln in out.split("\n")]
    cnt = collections.Counter(ln for ln in o_lines if len(ln) >= LINE_MIN_CHARS and _ALNUM.search(ln))
    runs, prev, run = collections.Counter(), None, 0
    for ln in (x for x in o_lines if x):
        run = run + 1 if ln == prev else 1
        prev = ln
        runs[ln] = max(runs[ln], run)
    s_units = collections.Counter(_source_units(src))
    label_max = collections.defaultdict(int)
    for u, c in s_units.items():
        lab = _label_core(u)
        label_max[lab] = max(label_max[lab], c)
    lines = []
    for ln, k in cnt.most_common():
        if k < LINE_MIN_COUNT:
            break
        lab = _label_core(ln)
        guard = max(s_units.get(ln, 0), label_max.get(lab, 0))
        lines.append({"kind": "line", "text": ln[:160], "count": k, "max_run": runs[ln], "label": lab,
                      "source_same_text": s_units.get(ln, 0), "source_max_same_label": label_max.get(lab, 0),
                      "flagged": k > guard})
    flagged = [x for x in subs + lines if x["flagged"]]
    return {"substring": subs, "lines": lines, "flagged": len(flagged),
            "flagged_substring": sum(1 for x in subs if x["flagged"]),
            "flagged_lines": sum(1 for x in lines if x["flagged"])}

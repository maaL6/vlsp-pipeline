from __future__ import annotations

import collections
import re
from decimal import Decimal, InvalidOperation

from .common import ANY_MARKER, _nfc, _ws

BIG_DIGITS = 4                # "big" numbers: integer part has >= 4 digits (money, years)

# ============================================================================ numbers
_NUM = re.compile(r"\d+(?:[.,]\d+)*")

# clause labels "1." "1-" "1)" "2.1." "(1)" at line start (optionally after quotes) are not numbers
_LABEL_START = re.compile(r"^(\s*[“\"'‘(\[]*\s*)(?:\(\d{1,3}\)|\d{1,3}(?:\.\d{1,2})*\s*[.)\-–]{1,2})(?=\s|$)")

# the same labels inside a flattened line, right after sentence punctuation ("...: 1. Abc")
_LABEL_MID = re.compile(r"(?<=[.;:!?”\"])(\s+)\d{1,3}(?:\.\d{1,2})*[.)\-–](?=\s+[^\W\d_“\"(])")

_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
           "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
           "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
           "sept": 9, "oct": 10, "nov": 11, "dec": 12}

_MONTH_ALWAYS = {k for k in _MONTHS if len(k) > 4 or k == "june" or k == "july"} - {"may"}

_MONTH_RE = re.compile(
    r"(?<![^\W\d_])(January|February|March|April|May|June|July|August|September|October|November|"
    r"December|Sept|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?(?![^\W\d_])", re.I)

_HEAD_ROMAN_NUM = re.compile(
    r"(?<![^\W\d_])((?:chapter|chương|section|mục|part|phần)[ \t]+)([IVXLC][IVXLCl]*)(?![^\W_])", re.I)

class NumItem(collections.namedtuple("NumItem", "value alt raw line")):
    """value: canonical string; alt: grouping reading of a decimal-shaped token (or None)."""

def _canon_int(digits):
    return str(int(digits))

def _canon_dec(int_digits, frac_digits):
    try:
        d = Decimal(f"{int(int_digits)}.{frac_digits}").normalize()
    except InvalidOperation:
        return f"{int(int_digits)}.{frac_digits}"
    return format(d, "f")

def _parse_token(tok, lang):
    """[(value, alt)] for one digit run. Grouping/decimal marks are read per language:
    en: ',' groups and '.' is decimal; vi: '.' groups and ',' is decimal. Unambiguous shapes
    (1,000,000 / 1.000.000 / 1,500.50 / 1.500,50 / 0.5 / 0,5) give the same value in both."""
    seps = [c for c in tok if c in ".,"]
    if not seps:
        return [(_canon_int(tok), None)]
    parts = re.split(r"[.,]", tok)
    if len(set(seps)) == 1:
        sep = seps[0]
        if len(parts) == 2:
            a, b = parts
            if len(b) == 3 and 1 <= len(a) <= 3 and a.strip("0"):
                grouped = _canon_int(a + b)
                if sep == ("," if lang == "en" else "."):
                    return [(grouped, None)]                 # native grouping
                return [(_canon_dec(a, b), grouped)]         # decimal, but maybe copied foreign grouping
            return [(_canon_dec(a, b), None)]                # 0.5 / 0,5 / 2.1 (dotted id)
        if 1 <= len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:]):
            return [(_canon_int("".join(parts)), None)]      # 1.000.000 / 1,000,000 / 04.000.000
        if sep == "." and len(parts) == 3 and len(parts[2]) == 4 and max(len(parts[0]), len(parts[1])) <= 2:
            return [(_canon_int(p), None) for p in parts]    # d.m.yyyy date
        if sep == ".":
            return [(".".join(_canon_int(p) for p in parts), None)]   # dotted id 1.2.3
        return [(_canon_int(p), None) for p in parts]        # "3,4,5" list
    pos = max(tok.rfind("."), tok.rfind(","))
    int_part, frac = tok[:pos], tok[pos + 1:]
    groups = re.split(r"[.,]", int_part)
    if ({c for c in int_part if c in ".,"} == {"," if tok[pos] == "." else "."}
            and 1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:])):
        return [(_canon_dec("".join(groups), frac), None)]  # 1,500.50 / 1.500,50
    return [(_canon_int(p), None) for p in re.split(r"[.,]", tok)]

def _roman_value(tok):
    t = tok.strip()
    if not t or not re.fullmatch(r"[IVXLCivxlc][IVXLCivxlcl]*", t):
        return None
    if t[0].isupper() and "l" in t[1:]:
        t = t[0] + t[1:].replace("l", "I")    # OCR "Vll" -> VII
    t = t.upper()
    if len(t) == 1 and t in "LCDM":
        return None                            # single L/C: a letter label, not a numeral
    if not re.fullmatch(r"M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})", t):
        return None
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(t):
        v = vals[ch]
        total += -v if i + 1 < len(t) and vals[t[i + 1]] > v else v
    return total or None

def _map_months(line):
    def repl(m):
        word = m.group(1)
        key = word.lower()
        if not (word[0].isupper()):
            return m.group(0)                  # "may", "march" (verbs) never mapped
        if key not in _MONTH_ALWAYS:           # May + abbreviations: only next to a number
            after = line[m.end():m.end() + 4]
            before = line[max(0, m.start() - 12):m.start()]
            if not (re.match(r"\s*\d", after) or re.search(r"\d(?:st|nd|rd|th)?\s+(?:of\s+)?$", before)):
                return m.group(0)
        return f" {_MONTHS[key]} "
    return _MONTH_RE.sub(repl, line)

def _roman_heading_sub(m):
    v = _roman_value(m.group(2))
    return m.group(1) + str(v) if v else m.group(0)

def number_items(text, lang):
    """All numbers of `text` as NumItem(value, alt, raw, line_no); clause labels skipped,
    month names and heading roman numerals mapped to numbers."""
    items = []
    for ln, line in enumerate(ANY_MARKER.sub(" ", _nfc(text)).split("\n"), 1):
        if not any(ch.isdigit() for ch in line) and not _MONTH_RE.search(line) \
                and not _HEAD_ROMAN_NUM.search(line):
            continue
        work = _LABEL_START.sub(lambda m: m.group(1), line)
        work = _LABEL_MID.sub(lambda m: m.group(1), work)
        work = _HEAD_ROMAN_NUM.sub(_roman_heading_sub, work)
        work = _map_months(work)
        for m in _NUM.finditer(work):
            for value, alt in _parse_token(m.group(0), lang):
                items.append(NumItem(value, alt, m.group(0), ln))
    return items

def number_tokens(text, lang):
    """Value-normalised multiset of the numbers in `text`: grouping per language
    (1,000,000 en == 1.000.000 vi), decimals (0.5 en == 0,5 vi), zero padding (02 == 2),
    month names (June 03, 1998 ~ ngày 03 tháng 6 năm 1998); clause labels are ignored."""
    return collections.Counter(i.value for i in number_items(text, lang))

def _is_big(value):
    return len(value.split(".")[0].lstrip("-")) >= BIG_DIGITS

def display_value(value):
    """Vietnamese-style display: 16000000 -> 16.000.000, 0.5 -> 0,5, years unchanged."""
    if "." in value:
        return value.replace(".", ",")
    return f"{int(value):,}".replace(",", ".") if len(value) >= 5 else value

def compare_numbers(src, out, src_lang, tgt_lang, context=False, max_context=3, src_items=None):
    """Multiset diff of number values source vs output (warning material, not proof of error).

    Exact value matches first; then a token whose reading is decimal but whose shape is a
    grouping in the other language (vi '500,000', en '500.000') may match that grouping value:
    such pairs are listed as `format_only` (format issue, not a value change)."""
    s_items = src_items if src_items is not None else number_items(src, src_lang)
    o_items = number_items(out, tgt_lang)
    sc = collections.Counter(i.value for i in s_items)
    oc = collections.Counter(i.value for i in o_items)
    common = sc & oc
    rem_s, rem_o = sc - common, oc - common
    format_only = []
    for items, rem_a, rem_b, side in ((o_items, rem_o, rem_s, "output"), (s_items, rem_s, rem_o, "source")):
        for it in items:
            if it.alt and rem_a[it.value] > 0 and rem_b[it.alt] > 0:
                rem_a[it.value] -= 1
                rem_b[it.alt] -= 1
                format_only.append({"side": side, "raw": it.raw, "read_as": it.value, "matched": it.alt,
                                    "line": it.line})
    rem_s, rem_o = +rem_s, +rem_o
    missing = [{"value": v, "src": sc[v], "out": oc[v], "diff": n} for v, n in sorted(rem_s.items(), key=_vkey)]
    extra = [{"value": v, "src": sc[v], "out": oc[v], "diff": n} for v, n in sorted(rem_o.items(), key=_vkey)]
    res = {"src_numbers": len(s_items), "out_numbers": len(o_items),
           "matched": sum(common.values()) + len(format_only),
           "missing_total": sum(rem_s.values()), "extra_total": sum(rem_o.values()),
           "missing": missing, "extra": extra, "format_only": format_only,
           "big_missing": [d for d in missing if _is_big(d["value"])],
           "big_extra": [d for d in extra if _is_big(d["value"])]}
    if context:
        s_lines, o_lines = _nfc(src).split("\n"), _nfc(out).split("\n")
        for d in res["big_missing"]:
            d["display"] = display_value(d["value"])
            d["src_context"] = _contexts(s_items, s_lines, d["value"], max_context)
            d["out_context"] = _contexts(o_items, o_lines, d["value"], max_context)
        for d in res["big_extra"]:
            d["display"] = display_value(d["value"])
            d["out_context"] = _contexts(o_items, o_lines, d["value"], max_context)
    return res

def _vkey(kv):
    v = kv[0]
    try:
        return (0, Decimal(v), v)
    except InvalidOperation:
        return (1, Decimal(0), v)

def _contexts(items, lines, value, limit):
    out, seen = [], set()
    for it in items:
        if (it.value == value or it.alt == value) and it.line not in seen:
            seen.add(it.line)
            text = _ws(lines[it.line - 1]) if it.line - 1 < len(lines) else ""
            out.append({"line": it.line, "raw": it.raw, "text": text[:220]})
            if len(out) >= limit:
                break
    return out

"""Translation units (SPEC section 3, units.py): long-node splitting, packing, markers, payload
rendering, output parsing, output budget and recovery units.

Works on the SOURCE side only (node payloads); never reads references.

Integration notes for callers (runtime):
- Call `split_long_nodes(nodes, tok, cfg)` (per-kind caps) or `(nodes, tok, cfg.source_cap)` before
  `pack_units`; pass the SAME post-split list to `pack_units` and build
  `nodes_by_id = {n.node_id: n for n in that_list}` (copy nodes included, list order kept).
- A node that still exceeds the cap of its unit type is packed alone as kind "single" (no markers).
- Split pieces `nX.1, nX.2 ...` partition `content_raw` of nX (first keeps lead, last keeps trail);
  assemble nX as `lead + " ".join(piece translations) + trail` (see `merge_child_segments`).
- `recovery_units(..., independent=False)` may split nodes again (`nX.1.1` ...); the new pieces are
  ADDED to `nodes_by_id` (caller's dict) so check_unit/assembly can look them up.
"""
# Split from lam_docmt/units.py (statements verbatim; tools/split_from_lam_docmt.py).
from __future__ import annotations

import re
import unicodedata as ud

from ..core.datatypes import Node
from ..core.text_normalize import HOMOGLYPHS, _INVIS, normalize

SIGNATURE_KINDS = ("signature", "signature_name")

# Characters normalize() may remove/replace: used to check that no "[SEG_" appears after normalization.
_FOLD = dict(HOMOGLYPHS)

_FOLD.update({ord(ch): None for ch in _INVIS + "­"})

# ------------------------------------------------------------------------------ payload helpers
_HYPHEN_END = re.compile(r"[^\W\d_]-$")

_SPACE_RUN = re.compile(r"[ \t]+")

def unwrap(text: str) -> str:
    """SPEC section 2: soft-wrapped lines joined with one space (no space after a line ending in
    letter + '-'), runs of spaces collapsed to one, ends stripped."""
    parts = []
    for line in text.split("\n"):
        s = _SPACE_RUN.sub(" ", line.strip())
        if not s:
            continue
        if parts and not _HYPHEN_END.search(parts[-1]):
            parts.append(" ")
        parts.append(s)
    return "".join(parts)

def payload_of(content_raw: str) -> str:
    """Model input for a raw content span: normalize(unwrap(content_raw))."""
    return normalize(unwrap(content_raw)).strip()

def root_node_id(node_id: str) -> str:
    """'n00042.1.2' -> 'n00042' (the parser node a split piece belongs to)."""
    return node_id.split(".", 1)[0]

# ----------------------------------------------------------------------------- long-node splitting
_OPENERS = "\"'“‘«(["

_CLOSERS = "\"'”’»)]"

_ABBREV = frozenset(a.lower() for a in (
    "No.", "Nos.", "Art.", "Arts.", "Dr.", "Mr.", "Mrs.", "Ms.", "Mt.", "St.", "e.g.", "i.e.", "etc.",
    "TT.", "QĐ.", "NĐ.", "Co.", "Ltd.", "Inc.", "Corp.", "Jr.", "Sr.", "vs.", "cf.", "approx.",
    "Vol.", "Vols.", "p.", "pp.", "para.", "paras.", "Cl.", "Ch.", "Sec.", "Dept.", "Gov.", "Govt.",
    "Prof.", "Fig.", "Ref.", "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.",
    "Oct.", "Nov.", "Dec.", "TP.", "Tp.", "ThS.", "TS.", "PGS.", "GS.", "KT.", "TM.", "Q.", "P."))

_MULTI_DOT_ABBREV = re.compile(r"(?:[^\W\d_]{1,3}\.){2,}")       # U.S., e.g., Ph.D.

_INITIAL = re.compile(r"[^\W\d_]\.")                               # V. (initial) / a. (point)

_LABEL = re.compile(r"\(?(?:\d{1,3}(?:\.\d{1,3})*|[^\W\d_]|[IVXLCDM]{1,6})[.)]-?")

_NUMBER = re.compile(r"\d[\d.,]*")

_REF_TARGET = re.compile(r"\(?(?:\d[\w./-]*|[IVXLCDM]{1,6}|[^\W\d_])[.,;:)]*")   # 5 / 12/2011/TT-BTP / II / a)

_UNIT_AFTER = frozenset("""
days day months month years year hours hour minutes minute seconds second weeks week working
workdays workday times persons person people members copies pages percent per cent %
vnd vnđ đ dong đồng usd eur us$ million millions billion billions thousand thousands
kg g mg km m m2 m3 m² ha hectares hectare tons ton tonnes tonne liters litres liter litre cm mm
ngày tháng năm giờ phút giây tuần lần người bản trang triệu tỷ tỉ nghìn ngàn tấn lít tuổi
""".split())

_CURRENCY_BEFORE = frozenset("vnd vnđ usd eur us$ $".split())

_REF_WORDS = frozenset("""
article articles art. arts. clause clauses point points item items chapter chapters section sections
part parts appendix appendices annex annexes table tables form forms decree decision circular law
resolution no. nos. number paragraph paragraphs subclause sub-clause
điều khoản điểm chương mục phần lục bảng mẫu số tiết ngày tháng năm
""".split())

_MONTHS = frozenset("""
january february march april may june july august september october november december
jan feb mar apr jun jul aug sep sept oct nov dec
""".split())

def _bare(tok: str) -> str:
    return tok.strip(_OPENERS + _CLOSERS)

def _is_number(tok: str) -> bool:
    return bool(_NUMBER.fullmatch(_bare(tok).rstrip(".,;:")))

def _is_abbrev(tok: str) -> bool:
    t = tok.lstrip(_OPENERS).rstrip(_CLOSERS).rstrip(",")
    tl = t.lower()
    return tl in _ABBREV or bool(_MULTI_DOT_ABBREV.fullmatch(t)) or bool(_INITIAL.fullmatch(t))

def _is_label(c: str, tok_start: int, tok: str) -> bool:
    """Numbering label ('1.', '2.1.', 'a)', 'II.') at content start / after ':;.' / at line start."""
    if not _LABEL.fullmatch(tok.rstrip(",")):
        return False
    before = c[:tok_start]
    stripped = before.rstrip()
    return (not stripped) or stripped[-1] in ".:;" or "\n" in before[len(stripped):]

def _is_glue(c: str, a: int, left: str, right: str) -> bool:
    """Whitespace that must not be cut: abbreviation / label / number+unit / currency+number /
    reference word+number / dates / number lists."""
    if _is_abbrev(left) or _is_label(c, a - len(left), left):
        return True
    lb, rb = _bare(left), _bare(right)
    l_low = lb.lower().rstrip(",;:")
    r_low = rb.lower().rstrip(".,;:")
    l_num, r_num = _is_number(left), _is_number(right)
    r_starts_digit = rb[:1].isdigit()
    if l_num and (r_low in _UNIT_AFTER or r_low in _MONTHS):
        return True
    if l_low in _UNIT_AFTER and r_low in _UNIT_AFTER:            # "working days", "triệu đồng"
        return True
    if l_low.rstrip(".") in _CURRENCY_BEFORE and r_starts_digit:  # "VND 5,000,000"
        return True
    if (l_low in _REF_WORDS or l_low.rstrip(".") in _MONTHS) and _REF_TARGET.fullmatch(rb):
        return True                                             # "Article 5", "point a", "May 5"
    if left.endswith(",") and left[:-1].rstrip(_CLOSERS)[-1:].isdigit() and r_starts_digit:
        return True                                             # "October 14, 2015", "1, 2"
    if l_num and re.fullmatch(r"\d{3}[.,;:)]*", rb):              # space-grouped thousands
        return True
    return False

def _tier(c: str, a: int, b: int):
    """Cut preference for the whitespace run c[a:b]: 0 sentence, 1 clause, 2 whitespace,
    3 whitespace inside a capitalised name run, 4 glue (last resort), None never."""
    if "\n" in c[a:b] and _HYPHEN_END.search(c[max(0, a - 2):a]):
        return None                                             # hyphenated line break: same word
    lm = re.search(r"\S+$", c[max(0, a - 80):a])
    rm = re.match(r"\S+", c[b:b + 80])
    if not lm or not rm:
        return None
    left, right = lm.group(0), rm.group(0)
    if _is_glue(c, a, left, right):
        return 4
    core = left.rstrip(_CLOSERS)
    last = core[-1:]
    nxt = c[b]
    if last in (".", "!", "?", ";", ":"):
        if nxt.isupper() or nxt.isdigit() or nxt in _OPENERS:
            return 0
        return 1
    if last == ",":
        return 1
    lb, rb = _bare(left), _bare(right)
    if lb[:1].isupper() and lb[-1:].isalpha() and rb[:1].isupper():
        return 3
    return 2

def _candidates(c: str):
    out = []
    for m in re.finditer(r"\s+", c):
        a, b = m.span()
        if a == 0 or b >= len(c):
            continue
        t = _tier(c, a, b)
        if t is not None:
            out.append((b, t))
    return out

def _fits(c: str, start: int, end: int, tok, cap: int) -> bool:
    return tok.count(payload_of(c[start:end])) <= cap

def _max_prefix(c: str, start: int, tok, cap: int) -> int:
    """Largest end (> start) with tokens(payload(c[start:end])) <= cap (binary search on chars)."""
    lo, hi = start + 1, len(c)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _fits(c, start, mid, tok, cap):
            lo = mid
        else:
            hi = mid - 1
    return lo

def _cut_positions(c: str, tok, cap: int):
    """[(cut, tier)] such that every piece's payload has <= cap tokens."""
    cands = _candidates(c)
    cuts = []
    start = 0
    while not _fits(c, start, len(c), tok, cap):
        limit = _max_prefix(c, start, tok, cap)
        window = [(b, t) for b, t in cands if start < b <= limit and c[start:b].strip()]
        chosen = None
        for tier in range(5):
            for b in sorted((b for b, t in window if t == tier), reverse=True):
                if _fits(c, start, b, tok, cap):
                    chosen = (b, tier)
                    break
            if chosen:
                break
        if chosen is None:                                      # no whitespace at all: hard cut
            p = max(limit, start + 1)
            while start + 1 < p < len(c) and ud.combining(c[p]):
                p -= 1
            chosen = (p, 5)
        if chosen[0] >= len(c):
            break
        cuts.append(chosen)
        start = chosen[0]
    return cuts

def split_node(node: Node, tok, cap: int) -> list:
    """Split one translatable node into pieces '<id>.1', '<id>.2' ... with payload <= cap tokens.

    Cut preference: sentence end ('. ' '; ' ': ' '! ' '? ' before upper-case/digit/quote), then
    clause (', ' or ';'/':' before lower-case), then whitespace nearest to the cap, avoiding cuts
    inside numbers, after abbreviations/labels, between number and unit/currency, inside dates and
    reference+number pairs ("Article 5"), inside capitalised name runs. Pieces partition
    content_raw exactly; the first keeps lead (and `protected`), the last keeps trail."""
    c = node.content_raw
    cuts = _cut_positions(c, tok, cap) if c.strip() else []
    if not cuts:
        return [node]
    bounds = [0] + [p for p, _ in cuts] + [len(c)]
    forced = {i for i, (_, t) in enumerate(cuts) if t >= 4}
    base = node.raw_start + len(node.lead)
    k = len(bounds) - 1
    pieces = []
    for j in range(k):
        a, b = bounds[j], bounds[j + 1]
        first, last = j == 0, j == k - 1
        content = c[a:b]
        flags = list(node.flags) + ["split_piece"]
        if (j in forced) or (j - 1 in forced):
            flags.append("split_forced")
        pieces.append(Node(
            node_id=f"{node.node_id}.{j + 1}", kind=node.kind,
            raw_start=node.raw_start if first else base + a,
            raw_end=node.raw_end if last else base + b,
            lead=node.lead if first else "", content_raw=content, trail=node.trail if last else "",
            payload=payload_of(content), translatable=True, scope_id=node.scope_id,
            path=list(node.path), confidence=node.confidence,
            protected=dict(node.protected) if first else {},
            table=dict(node.table) if node.table is not None else None, flags=flags))
    return pieces

def _run_key(n: Node):
    if n.table is not None or n.kind == "table_cell":
        return ("table", n.scope_id, (n.table or {}).get("group"))
    if n.kind in SIGNATURE_KINDS:
        return ("signature", n.scope_id, None)
    return ("text", n.scope_id, None)

def _limits(kind: str, cfg):
    """(token cap incl. markers, node limit) of a unit kind."""
    if kind == "table":
        return cfg.table_cap, cfg.max_table_nodes
    if kind == "signature":
        return cfg.signature_cap, cfg.max_nodes
    return cfg.source_cap, cfg.max_nodes

def _cap_for(n: Node, cap) -> int:
    if isinstance(cap, int):
        return cap
    return _limits(_run_key(n)[0], cap)[0]                    # RunConfig: per-kind caps

def split_long_nodes(nodes, tok, cap) -> list:
    """Replace every translatable node whose payload has more than `cap` tokens by its pieces.

    `cap` is an int (same cap for all nodes) or a RunConfig (source_cap for text nodes, table_cap
    for table cells, signature_cap for signature nodes). Copy nodes are never split."""
    out = []
    for n in nodes:
        c = _cap_for(n, cap)
        if not n.translatable or tok.count(n.payload) <= c:
            out.append(n)
        else:
            out.extend(split_node(n, tok, c))
    return out

# ------------------------------------------------------------------------------------- packing
def _check_partition(nodes):
    prev = None
    for n in nodes:
        if n.raw_end - n.raw_start != len(n.raw):
            raise ValueError(f"node {n.node_id}: raw offsets do not match lead+content_raw+trail")
        if prev is not None and n.raw_start != prev.raw_end:
            raise ValueError(f"nodes {prev.node_id} -> {n.node_id}: offsets not contiguous")
        prev = n

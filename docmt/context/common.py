"""Source context, scoped memory and soft glossary (SPEC section 3 "context.py"; draft section 2).

Evidence for one unit, in priority order:
  1. title / structural path of the unit              origin source_title, source_path
  2. definitions / references from the same source    origin source_definition, source_reference
  3. <= cfg.soft_terms train-glossary terms present in the unit source, only if cfg.soft_glossary
                                                       origin train_glossary
  4. <= cfg.recent_accepted_units accepted units of the same doc + scope, only if cfg.target_history
                                                       origin accepted_history
Group 1-2 only if cfg.source_context. The rendered background is <= cfg.evidence_cap HyTok tokens:
items are taken in priority order and an item that does not fit beside the higher-priority items is
dropped whole (lowest priority is dropped first); only the title is truncated (tok.truncate_tokens).

Inputs are the SOURCE side of one document, the train glossary and accepted runtime outputs. Nothing
here reads reference/gold text. No sample id, term, number or article number is hard-coded: every
rule below is a generic pattern of Vietnamese/English legal drafting.
"""
# Split from lam_docmt/context.py (statements verbatim; tools/split_from_lam_docmt.py).
from __future__ import annotations

import re

ORIGINS = ("source_title", "source_path", "source_definition", "source_reference",
           "train_glossary", "accepted_history")

KIND_OF_ORIGIN = {"source_title": "title_path", "source_path": "title_path",
                  "source_definition": "definition", "source_reference": "reference",
                  "train_glossary": "glossary", "accepted_history": "history"}

# Tunables; override with cfg.extra[<key>].
DEFAULTS = {
    "context_title_cap": 96,          # tokens; the title is the only item that may be truncated
    "context_max_definitions": 4,     # definitions per unit (most specific first)
    "context_max_references": 4,      # referenced article headings per unit (first mention first)
    "context_history_item_cap": 160,  # tokens of one accepted-history excerpt (tail of the unit)
}

# Excerpt bounds applied when the index is built (an excerpt ends with "…" when bounded).
HEADING_WORDS = 32

LEAD_WORDS = 40          # clause lead-in / quote lead-in shown in the path

DEF_WORDS = 80           # one definition item

ANTECEDENT_WORDS = 40    # words before "(hereinafter referred to as X)" kept in the excerpt

MAX_TERM_WORDS = {"en": 15, "vi": 20}

HEADING_KINDS = ("part", "chapter", "section", "article")

ELLIPSIS = "…"

_MARKER_RE = re.compile(r"\[SEG[0-9A-F]{0,4}_\d{6}\]")

_APOS = "'’‘`ʼ"

_HYPH = "-‐‑–"

_FOLD = {ord(c): "'" for c in _APOS}

_FOLD.update({ord(c): "-" for c in _HYPH})

_VI_CHARS = set("ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ")

def _fold(s: str) -> str:
    """Matching key: lower case, one apostrophe / hyphen form, single spaces."""
    return " ".join(s.lower().translate(_FOLD).split())

def _term_regex(term: str):
    """Case-insensitive, word-bounded pattern; apostrophe/hyphen variants and spacing tolerated."""
    out = []
    for ch in " ".join(term.split()):
        if ch == " ":
            out.append(r"\s+")
        elif ch in _APOS:
            out.append("[" + re.escape(_APOS) + "]")
        elif ch in _HYPH:
            out.append("[" + re.escape(_HYPH) + "]")
        else:
            out.append(re.escape(ch))
    return re.compile(r"(?<!\w)" + "".join(out) + r"(?!\w)", re.IGNORECASE)

def _bound_words(text: str, max_words: int, prefer_sentence: bool = False) -> str:
    """Excerpt of at most max_words words (first sentence when it fits); "…" marks a cut."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    if prefer_sentence:
        for m in re.finditer(r"[.;:](?=\s)", text):
            head = text[:m.end()]
            n = len(head.split())
            if n > max_words:
                break
            if n >= 6 and not _ABBR_END.search(text[:m.end()]):
                return " ".join(head.split()) + " " + ELLIPSIS
    return " ".join(words[:max_words]) + " " + ELLIPSIS

def _overlaps(spans_a, spans_b) -> bool:
    for s1, e1 in spans_a:
        for s2, e2 in spans_b:
            if s1 < e2 and s2 < e1:
                return True
    return False

def _strip_markers(text: str) -> list:
    """Segments of a payload/output: split on internal markers, else on lines. Markers are removed."""
    text = text or ""
    if _MARKER_RE.search(text):
        parts = _MARKER_RE.split(text)
    else:
        parts = text.split("\n")
    return [" ".join(p.split()) for p in parts if p.strip()]

def guess_lang(text: str) -> str:
    """Fallback when no SourceIndex is given: Vietnamese if Vietnamese-only letters occur."""
    return "vi" if any(c in _VI_CHARS for c in (text or "").lower()) else "en"

def _is_quote_scope(scope_id: str) -> bool:
    return str(scope_id or "").startswith("quote")

def _starts_with_quote(text: str) -> bool:
    return text.lstrip()[:1] in ("“", '"', "‘")

def _mostly_upper(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) >= 0.8 * len(letters)

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}

def _canon_num(num: str) -> str:
    s = str(num).strip().rstrip(".:-–)").lower()
    m = re.fullmatch(r"0*(\d+)([a-zđ]?)", s)
    if m:
        return str(int(m.group(1))) + m.group(2)
    if s and all(c in _ROMAN for c in s):
        total, prev = 0, 0
        for c in reversed(s):
            v = _ROMAN[c]
            total = total - v if v < prev else total + v
            prev = max(prev, v)
        return str(total)
    return s

_LABEL_HEAD = re.compile(
    r"^\W*(article|điều|chapter|chương|section|mục|part|phần)\s+([0-9]+[a-zđ]?|[ivxlcdm]+)(?!\w)",
    re.IGNORECASE)

_KIND_WORD = {"article": "article", "điều": "article", "chapter": "chapter", "chương": "chapter",
              "section": "section", "mục": "section", "part": "part", "phần": "part"}

_LABEL_ONLY = re.compile(
    r"^\W*(?:article|điều|chapter|chương|section|mục|part|phần)\s+(?:[0-9]+[a-zđ]?|[ivxlcdm]+)"
    r"\s*[.:\-–]*\s*$", re.IGNORECASE)

def _canon_label(label) -> tuple:
    s = str(label).strip()
    m = _LABEL_HEAD.match(s)
    if m:
        return (_KIND_WORD[m.group(1).lower()], _canon_num(m.group(2)))
    m = re.fullmatch(r"\(?(\d+(?:\.\d+)*)[.)\-/]*", s)
    if m:
        return ("clause", m.group(1))
    m = re.fullmatch(r"\(?([a-zđ]{1,2})[).]?", s, re.IGNORECASE)
    if m:
        return ("point", m.group(1).lower())
    return ("other", s.lower())

def _own_label(node) -> Optional[tuple]:
    """Canonical label of a heading / clause / point node (protected first, then payload)."""
    prot = node.protected or {}
    ht, hn = prot.get("heading_type"), prot.get("heading_number")
    if ht and hn is not None and str(ht).lower() in ("article", "chapter", "section", "part"):
        return (str(ht).lower(), _canon_num(hn))
    if node.kind in HEADING_KINDS:
        m = _LABEL_HEAD.match(node.payload or "")
        if m and _KIND_WORD[m.group(1).lower()] == node.kind:
            return (node.kind, _canon_num(m.group(2)))
        return None
    if node.kind in ("clause", "point"):
        lab = prot.get("label")
        if lab:
            return _canon_label(lab)
        m = re.match(r"^\s*\(?(\d+(?:\.\d+)*)[.)\-/]", node.payload or "")
        if node.kind == "clause" and m:
            return ("clause", m.group(1))
        m = re.match(r"^\s*\(?([a-zđ]{1,2})[).]", node.payload or "", re.IGNORECASE)
        if node.kind == "point" and m:
            return ("point", m.group(1).lower())
    return None

def _canon_path(node) -> list:
    return [_canon_label(p) for p in (node.path or [])]

_ABBR_END = re.compile(
    r"(?:\b(?:No|Nos|Art|Arts|Dr|Mr|Mrs|Ms|St|Co|Ltd|Inc|Jr|Sr|etc|vs|cf|TT|QĐ|NĐ|Tp|TP)\.|"
    r"\b(?:e\.g|i\.e)\.|\b[A-ZĐ]\.|\d\.)$")

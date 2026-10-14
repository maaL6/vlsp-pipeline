from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field, replace

from .common import ANTECEDENT_WORDS, DEF_WORDS, ELLIPSIS, HEADING_KINDS, HEADING_WORDS, LEAD_WORDS, MAX_TERM_WORDS, _ABBR_END, _LABEL_HEAD, _LABEL_ONLY, _bound_words, _canon_num, _canon_path, _fold, _is_quote_scope, _mostly_upper, _overlaps, _own_label, _starts_with_quote, _term_regex
from ..core.datatypes import EvidenceItem

_DEF_HEAD = {
    "en": re.compile(r"\b(?:interpretation\s+of\s+(?:terms?|words|expressions)|definitions?|"
                     r"terms?\s+used|terms?\s+interpretation|terms?\s+and\s+concepts?|terminology|"
                     r"explanation\s+of\s+terms?|glossary)\b", re.IGNORECASE),
    "vi": re.compile(r"(?<!\w)(?:giải\s+thích\s+(?:từ\s+ngữ|thuật\s+ngữ)|thuật\s+ngữ|khái\s+niệm|"
                     r"định\s+nghĩa)(?!\w)", re.IGNORECASE),
}

_DEF_INTRO = {
    "en": re.compile(r"\b(?:construed|understood|interpreted|defined)\s+as\s+follows?\b|"
                     r"\bhave\s+the\s+following\s+meanings?\b", re.IGNORECASE),
    "vi": re.compile(r"(?<!\w)(?:hiểu\s+như\s+sau|giải\s+thích\s+như\s+sau|có\s+nghĩa\s+như\s+sau)",
                     re.IGNORECASE),
}

_ITEM_LABEL = re.compile(r"^\s*(?:\(?\d+(?:\.\d+)*[.)/\-]+(?!\d)|\(?[a-zđ]{1,2}[).](?=\s)|[-+•*–])\s*")

# Header-block lines that a parser may tag as "title" (number, motto, place/date): never the title.
_HEADER_LINE = re.compile(
    r"^\W*(?:No|Nos|Số)\s*[.:]|^\W*(?:socialist\s+republic\s+of\s+viet\s*nam|"
    r"cộng\s+hòa\s+xã\s+hội\s+chủ\s+nghĩa\s+việt\s+nam|independence\s*[-–]\s*freedom|"
    r"độc\s+lập\s*[-–]\s*tự\s+do)", re.IGNORECASE)

_DATE_LINE = re.compile(
    r"\b(?:1[89]|20)\d\d\b.*$|ngày\s+\d{1,2}\s+tháng\s+\d{1,2}", re.IGNORECASE)

def _is_header_line(text: str) -> bool:
    if _HEADER_LINE.search(text):
        return True
    words = text.split()
    return len(words) <= 10 and bool(_DATE_LINE.search(text)) and (
        "," in text or "ngày" in text.lower())

_QUOTED_TERM = re.compile(
    r'^(?:(?:the|a|an)\s+)?(?:(?:term|terms|expression|phrase|word|concept|thuật\s+ngữ|từ\s+ngữ|'
    r'cụm\s+từ|từ|khái\s+niệm)\s+)?[“"]\s*([^“”"]{2,160}?)\s*[”"]', re.IGNORECASE)

_DEF_VERB = {
    "en": re.compile(r"\s(?:means|mean|refers\s+to|refer\s+to|shall\s+mean|is\s+defined\s+as|"
                     r"are\s+defined\s+as|shall\s+be\s+(?:construed|understood)\s+as|"
                     r"(?:is|are)\s+understood\s+as|denotes|denote|includes|include|is|are)\s",
                     re.IGNORECASE),
    "vi": re.compile(r"\s(?:là|dùng\s+để\s+chỉ|được\s+hiểu\s+là|có\s+nghĩa\s+là|bao\s+gồm|gồm)\s",
                     re.IGNORECASE),
}

_TERM_TAIL = re.compile(
    r"\s+(?:referred\s+to\s+herein|(?:as\s+)?(?:mentioned|used|defined|specified)\s+(?:herein|"
    r"in\s+this\s+\w+(?:\s+\w+)?)|quy\s+định\s+tại\s+\w+(?:\s+\w+)?\s+này|nêu\s+tại\s+\w+(?:\s+\w+)?"
    r"\s+này|trong\s+\w+(?:\s+\w+)?\s+này)\s*$", re.IGNORECASE)

_FUNC_END = {
    "en": {"and", "or", "of", "the", "a", "an", "in", "for", "to", "by", "with", "on", "at", "from",
           "that", "which", "who", "whose", "as", "than", "such", "its", "their", "this", "these",
           "is", "are", "be", "not"},
    "vi": {"và", "hoặc", "của", "cho", "với", "tại", "theo", "để", "mà", "thì", "là", "các",
           "những", "được", "bị", "trong", "về", "từ", "không"},
}

_FUNC_START = {
    "en": {"in", "for", "under", "within", "according", "pursuant", "where", "when", "if", "it",
           "there", "this", "these", "each", "any", "all", "no", "not", "with", "by", "to", "on",
           "at", "from", "as", "after", "before", "upon", "unless", "except"},
    "vi": {"trong", "theo", "tại", "khi", "nếu", "căn", "đối", "với", "mọi"},
}

_INLINE = {
    "en": [
        re.compile(r"\(\s*(?:here(?:in)?after|hereunder)\b\s*,?\s*"
                   r"(?:(?:collectively|jointly|together|generally|commonly|simply|briefly)\s+)?"
                   r"(?:(?:referred|called|abbreviated|shortened|known|named|termed|designated|cited)\s+)?"
                   r"(?:(?:collectively|jointly|together|generally|commonly|simply|briefly)\s+)?"
                   r"(?:to\s+)?(?:as\s+)?([^()]{2,200}?)\s*\)", re.IGNORECASE),
        re.compile(r"\(\s*(?:(?:collectively|jointly)\s+)?(?:referred\s+to|called|abbreviated|known)"
                   r"\s+(?:collectively\s+)?as\s+([^()]{2,200}?)\s*\)", re.IGNORECASE),
    ],
    "vi": [
        re.compile(r"\(\s*(?:sau\s+đây\s+)?(?:được\s+)?(?:gọi|viết)\s+(?:tắt\s+|chung\s+)?(?:tắt\s+)?"
                   r"là\s*:?\s*([^()]{2,200}?)\s*\)", re.IGNORECASE),
    ],
}

_INLINE_SPLIT = {
    "en": re.compile(r"(?:,\s*|\s+)and\s+(?:abbreviated|shortened)\s+(?:as|to)\s+", re.IGNORECASE),
    "vi": re.compile(r"(?:,\s*|\s+và\s+)(?:được\s+)?viết\s+tắt\s+là\s+", re.IGNORECASE),
}

_QUOTED_ANY = re.compile(r"[“\"]\s*([^“”\"]{2,160}?)\s*[”\"]")

def _clean_term(term: str) -> str:
    t = " ".join(term.split()).strip(" ,;:.“”\"'‘’")
    t = re.sub(r"^(?:the|a|an)\s+", "", t, flags=re.IGNORECASE)
    t = _TERM_TAIL.sub("", t)
    return t.strip(" ,;:“”\"'‘’")

def _valid_term(term: str, lang: str, check_function_words: bool) -> bool:
    if len(term) < 3 or not any(c.isalpha() for c in term):
        return False
    words = term.split()
    if len(words) > MAX_TERM_WORDS.get(lang, 15):
        return False
    if check_function_words:
        if words[0].lower() in _FUNC_START.get(lang, ()):
            return False
        if words[-1].lower().strip(",") in _FUNC_END.get(lang, ()):
            return False
        if ";" in term:
            return False
    return True

def _paren_depth_ok(text: str, pos: int) -> bool:
    return text.count("(", 0, pos) <= text.count(")", 0, pos)

def _article_item_term(text: str, lang: str) -> Optional[str]:
    """Term defined by one definition item: `"X" means ...`, `X là ...`, `X: ...` (label stripped)."""
    m = _QUOTED_TERM.match(text)
    if m:
        t = _clean_term(m.group(1))
        return t if _valid_term(t, lang, False) else None
    cands = []
    for vm in _DEF_VERB[lang].finditer(text):
        if vm.start() > 240:
            break
        if _paren_depth_ok(text, vm.start()):
            cands.append(vm.start())
    cm = re.search(r":\s", text[:200])
    if cm and _paren_depth_ok(text, cm.start()):
        cands.append(cm.start())
    for pos in sorted(set(cands)):
        t = _clean_term(text[:pos])
        if _valid_term(t, lang, True):
            return t
    return None

def _sentence_start(text: str, end: int) -> int:
    start = 0
    for m in re.finditer(r"[.;:!?][”\"’)]?\s+", text[:end]):
        if text[m.start()] == "." and _ABBR_END.search(text[:m.start() + 1]):
            continue
        start = m.end()
    return start

def _inline_terms(raw: str, lang: str) -> list:
    raw = raw.strip()
    quoted = [q for q in _QUOTED_ANY.findall(raw)]
    if quoted:
        terms = quoted
    else:
        terms = _INLINE_SPLIT[lang].split(raw)
    out = []
    for t in terms:
        t = _clean_term(t)
        if _valid_term(t, lang, False) and t not in out:
            out.append(t)
    return out

_REF = {
    "en": re.compile(r"(?<!\w)(?:Articles?|ARTICLES?|articles?)\s+(\d+[a-zA-Z]?)(?!\w)"
                     r"((?:\s*(?:,\s*(?:and|or)?|;|and|or|to|through|-|–)\s*"
                     r"(?:(?:Articles?|articles?)\s+)?\d+[a-zA-Z]?(?!\w))*)"),
    "vi": re.compile(r"(?<!\w)(?:[Đđ]iều|ĐIỀU)\s+(\d+[a-zđ]?)(?!\w)"
                     r"((?:\s*(?:,\s*(?:và|hoặc)?|;|và|hoặc|đến|-|–)\s*"
                     r"(?:(?:[Đđ]iều|ĐIỀU)\s+)?\d+[a-zđ]?(?!\w))*)"),
}

_NUM = re.compile(r"\d+[a-zA-Zđ]?(?!\w)")

_DOC_WORDS_EN = {"law", "laws", "code", "codes", "decree", "decrees", "circular", "circulars",
                 "decision", "decisions", "ordinance", "ordinances", "resolution", "resolutions",
                 "constitution", "convention", "agreement", "treaty", "directive", "regulation",
                 "regulations", "rules", "charter", "protocol", "instruction", "instructions",
                 "statute", "act", "covenant", "dispatch", "document"}

_STOP_SCAN_EN = {"with", "to", "for", "by", "accordance", "and", "or", "which", "that", "where",
                 "when", "if", "as", "shall", "must", "may", "is", "are"}

_DOC_WORDS_VI = re.compile(
    r"^(?:(?:của|tại|thuộc|theo)\s+)?(bộ\s+luật|luật|nghị\s+định|thông\s+tư\s+liên\s+tịch|thông\s+tư|"
    r"quyết\s+định|pháp\s+lệnh|nghị\s+quyết|hiến\s+pháp|công\s+ước|hiệp\s+định|điều\s+ước|quy\s+chế|"
    r"quy\s+định|chỉ\s+thị|văn\s+bản|điều\s+lệ|thỏa\s+thuận|thoả\s+thuận|hướng\s+dẫn|công\s+văn|"
    r"hợp\s+đồng)(?!\w)", re.IGNORECASE)

_AMEND_AFTER = {
    "en": re.compile(r"\b(?:is|are|shall\s+be|was|were|to\s+be|be|has\s+been|have\s+been)\s+"
                     r"(?:hereby\s+|herein\s+|now\s+)?(?:amended|supplemented|replaced|repealed|"
                     r"annulled|abolished|added|removed|deleted|corrected|revised|modified|rephrased|"
                     r"inserted|cancell?ed|invalidated)\b|\bas\s+follows\b", re.IGNORECASE),
    "vi": re.compile(r"(?:sửa\s+đổi|bổ\s+sung|thay\s+thế|bãi\s+bỏ|hủy\s+bỏ|huỷ\s+bỏ|đính\s+chính|"
                     r"như\s+sau)", re.IGNORECASE),
}

_AMEND_BEFORE = {
    "en": re.compile(r"\b(?:amend\w*|supplement\w*|replac\w*|repeal\w*|annul\w*|abolish\w*|"
                     r"insert\w*|add|adds|added|adding|addition|correct\w*|rectif\w*|invalidat\w*|"
                     r"cancel\w*)\b", re.IGNORECASE),
    "vi": re.compile(r"(?:sửa\s+đổi|bổ\s+sung|thay\s+thế|bãi\s+bỏ|hủy\s+bỏ|huỷ\s+bỏ|đính\s+chính|"
                     r"vào\s+sau|vào\s+trước)", re.IGNORECASE),
}

def _qualifier(after: str, lang: str):
    """("internal"|"external", doc_word or None) for the text right after an article reference."""
    s = after.lstrip(" ,")
    if lang == "vi":
        m = _DOC_WORDS_VI.match(s)
        if not m:
            return "internal", None
        rest = s[m.end():]
        if re.match(r"\s*(?:liên\s+tịch\s+)?này(?!\w)", rest, re.IGNORECASE):
            return "internal", " ".join(m.group(1).lower().split())
        return "external", None
    if re.match(r"(?:hereof|herein|hereunder|hereto|above|below)\b", s, re.IGNORECASE):
        return "internal", None
    m = re.match(r"(?:of|in|under|at)\s+", s, re.IGNORECASE)
    if not m:
        return "internal", None
    words = s[m.end():].split()[:9]
    for i, w in enumerate(words):
        wl = w.strip("’'\".,;:()").lower()
        if wl in _DOC_WORDS_EN:
            prev = words[i - 1].lower() if i else ""
            if prev in ("this", "these", "present"):
                return "internal", wl
            return "external", None
        if wl in _STOP_SCAN_EN:
            break
    return "internal", None

@dataclass
class _Heading:
    node_id: str
    kind: str               # part | chapter | section | article | clause
    number: str
    scope: str
    top: str
    pos: int
    text: str               # bounded excerpt of the heading (+ title line for part/chapter/section)
    spans: list = field(default_factory=list)

@dataclass
class _Definition:
    term: str
    key: str
    text: str
    node_id: str
    scope: str
    top: str
    pos: int
    span: tuple
    source: str             # "definition_article" | "inline"

class SourceIndex:
    """Index of ONE document's source, built before translation (titles, structural headings,
    definitions, article headings for references). Never sees any target or reference text."""

    def __init__(self):
        self.src_lang = "en"
        self.doc_id = None
        self.nodes = []
        self._pos = {}
        self._top_of = {}
        self._quoted = {}
        self._depth = {}
        self._titles = {}
        self._title_doc_word = {}
        self._headings = {}
        self._by_label = {}
        self._quote_chain = {}
        self.articles = {}
        self.ambiguous_articles = set()
        self.definitions = []
        self._defs_by_key = {}
        self._rx_cache = {}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, nodes, src_lang, doc_id=None) -> "SourceIndex":
        self = cls()
        self.src_lang = "vi" if str(src_lang).lower().startswith("vi") else "en"
        self.doc_id = doc_id
        self.nodes = list(nodes)
        self._pos = {n.node_id: i for i, n in enumerate(self.nodes)}
        self._build_scopes()
        self._build_quotes()
        self._build_headings()
        self._build_titles()
        self._build_quote_chains()
        self._build_definitions()
        return self

    def _text_nodes(self):
        return [(i, n) for i, n in enumerate(self.nodes) if n.translatable and (n.payload or "").strip()]

    def top_of(self, scope_id) -> str:
        return self._top_of.get(scope_id or "main", scope_id or "main")

    def _build_scopes(self):
        last_top = "main"
        for n in self.nodes:
            sc = n.scope_id or "main"
            if _is_quote_scope(sc):
                self._top_of.setdefault(sc, last_top)
            else:
                self._top_of.setdefault(sc, sc)
                last_top = sc

    def _build_quotes(self):
        """Curly-quote depth at the start of each node (reset at unquoted headings and top-scope
        changes), so quoted amendment text is not read as this document's own articles."""
        depth, cur_top = 0, None
        for _, n in self._text_nodes():
            top = self.top_of(n.scope_id)
            if top != cur_top:
                depth, cur_top = 0, top
            if n.kind in HEADING_KINDS and not _starts_with_quote(n.payload):
                depth = 0
            self._depth[n.node_id] = depth
            self._quoted[n.node_id] = (depth > 0 or _is_quote_scope(n.scope_id)
                                       or (n.kind in HEADING_KINDS and _starts_with_quote(n.payload)))
            depth = max(0, depth + n.payload.count("“") - n.payload.count("”"))

    def _heading_text(self, i, n) -> tuple:
        text, spans = n.payload, [(n.raw_start, n.raw_end)]
        if n.kind in ("part", "chapter", "section") and _LABEL_ONLY.match(n.payload):
            for j in range(i + 1, min(i + 6, len(self.nodes))):
                m = self.nodes[j]
                if (m.scope_id or "main") != (n.scope_id or "main"):
                    break
                if not m.translatable or not (m.payload or "").strip():
                    continue
                ok = (m.kind == "title"
                      or (m.kind in ("part", "chapter", "section") and not _LABEL_HEAD.match(m.payload))
                      or (m.kind in ("paragraph", "unknown") and len(m.payload.split()) <= 25
                          and _mostly_upper(m.payload)))
                if ok:
                    text = n.payload.rstrip() + " " + m.payload
                    spans.append((m.raw_start, m.raw_end))
                break
        return _bound_words(text, HEADING_WORDS), spans

    def _build_headings(self):
        for i, n in self._text_nodes():
            if n.kind not in HEADING_KINDS and n.kind != "clause":
                continue
            own = _own_label(n)
            if own is None:
                continue
            scope = n.scope_id or "main"
            path = _canon_path(n)
            if not path or path[-1] != own:
                path.append(own)
            if n.kind == "clause":
                text, spans = _bound_words(n.payload, LEAD_WORDS), [(n.raw_start, n.raw_end)]
            else:
                text, spans = self._heading_text(i, n)
            h = _Heading(n.node_id, own[0], own[1], scope, self.top_of(scope), i, text, spans)
            self._headings.setdefault((scope, tuple(path)), h)
            self._by_label.setdefault((scope, own), []).append(h)
            if own[0] == "article" and not self._quoted.get(n.node_id) and not _is_quote_scope(scope):
                key = (self.top_of(scope), own[1])
                if key in self.articles:
                    self.ambiguous_articles.add(key)
                else:
                    self.articles[key] = h

    def _build_titles(self):
        runs = {}
        state = {}
        for _, n in self._text_nodes():
            scope = n.scope_id or "main"
            if _is_quote_scope(scope):
                continue
            st = state.setdefault(scope, {"started": False, "done": False, "skipped": 0})
            if st["done"]:
                continue
            if n.kind in ("title", "attachment_title") and not st["started"] and _is_header_line(n.payload):
                continue
            if n.kind in ("title", "attachment_title"):
                runs.setdefault(scope, []).append(n)
                st["started"] = True
            elif n.kind == "note" and st["started"]:
                runs[scope].append(n)
            elif n.kind in ("header_cell", "separator"):
                continue
            elif st["started"] or n.kind in HEADING_KINDS or n.kind in ("clause", "point"):
                st["done"] = True
            else:
                st["skipped"] += 1
                if st["skipped"] > 8:
                    st["done"] = True
        for scope, run in runs.items():
            text = " ".join(" ".join(x.payload.split()) for x in run)
            self._titles[scope] = {"text": text, "node_ids": [x.node_id for x in run],
                                   "spans": [(x.raw_start, x.raw_end) for x in run]}
            first = run[0].payload.lower().split()
            self._title_doc_word[scope] = " ".join(first[:2]) if self.src_lang == "vi" else (
                first[0].strip(".:,") if first else "")

    def _resolve_chain(self, scope, cpath) -> list:
        out = []
        for k in range(1, len(cpath) + 1):
            if cpath[k - 1][0] not in HEADING_KINDS and cpath[k - 1][0] != "clause":
                continue
            h = self._headings.get((scope, tuple(cpath[:k])))
            if h is None:
                cands = self._by_label.get((scope, cpath[k - 1]), [])
                h = cands[0] if len(cands) == 1 else None
            if h is not None and h not in out:
                out.append(h)
        return out

    def _build_quote_chains(self):
        """For quote scopes: enclosing headings in the top scope + the lead-in line before the quote."""
        first_pos = {}
        for i, n in self._text_nodes():
            if _is_quote_scope(n.scope_id):
                first_pos.setdefault(n.scope_id, i)
        for q, i0 in first_pos.items():
            prev = None
            for j in range(i0 - 1, -1, -1):
                m = self.nodes[j]
                if m.translatable and (m.payload or "").strip() and not _is_quote_scope(m.scope_id):
                    prev = (j, m)
                    break
            chain = []
            if prev is not None:
                j, m = prev
                cp = _canon_path(m)
                own = _own_label(m)
                if own and (not cp or cp[-1] != own):
                    cp.append(own)
                chain = self._resolve_chain(m.scope_id or "main", cp)
                if m.kind not in HEADING_KINDS and all(h.node_id != m.node_id for h in chain):
                    chain.append(_Heading(m.node_id, "lead", "", m.scope_id or "main",
                                          self.top_of(m.scope_id), j,
                                          _bound_words(m.payload, LEAD_WORDS),
                                          [(m.raw_start, m.raw_end)]))
            self._quote_chain[q] = chain

    def _build_definitions(self):
        lang = self.src_lang
        in_block, block_scope = False, None
        for i, n in self._text_nodes():
            scope = n.scope_id or "main"
            payload = n.payload
            if block_scope is not None and self.top_of(scope) != self.top_of(block_scope):
                in_block = False
            if n.kind in HEADING_KINDS:
                head = _LABEL_HEAD.sub("", payload, count=1).lstrip(" .:-–")
                head = " ".join(head.split()[:12])
                in_block = (n.kind == "article" and bool(_DEF_HEAD[lang].search(head)))
                block_scope = scope
            elif _DEF_INTRO[lang].search(payload) and len(payload.split()) <= 40:
                in_block, block_scope = True, scope
            elif in_block and n.kind in ("clause", "point", "paragraph", "list_item", "unknown"):
                text = _ITEM_LABEL.sub("", payload, count=1).strip()
                term = _article_item_term(text, lang)
                if term:
                    self._add_def(term, _bound_words(text, DEF_WORDS, prefer_sentence=True), n, i,
                                  "definition_article")
            for rx in _INLINE[lang]:
                for m in rx.finditer(payload):
                    a = _sentence_start(payload, m.start())
                    ante = payload[a:m.start()].split()
                    if not ante:
                        continue
                    prefix = ""
                    if len(ante) > ANTECEDENT_WORDS:
                        ante = ante[-ANTECEDENT_WORDS:]
                        prefix = ELLIPSIS + " "
                    excerpt = prefix + " ".join(ante) + " " + " ".join(m.group(0).split())
                    for term in _inline_terms(m.group(1), lang):
                        self._add_def(term, excerpt, n, i, "inline")

    def _add_def(self, term, text, node, pos, source):
        key = _fold(term)
        for d in self._defs_by_key.get(key, []):
            if d.node_id == node.node_id and d.text == text:
                return
        d = _Definition(term, key, text, node.node_id, node.scope_id or "main",
                        self.top_of(node.scope_id), pos, (node.raw_start, node.raw_end), source)
        self.definitions.append(d)
        self._defs_by_key.setdefault(key, []).append(d)

    def _rx(self, key, term):
        rx = self._rx_cache.get(key)
        if rx is None:
            rx = self._rx_cache[key] = _term_regex(term)
        return rx

    # ------------------------------------------------------------------ queries
    def summary(self) -> dict:
        kinds = Counter(d.source for d in self.definitions)
        return {"doc_id": self.doc_id, "src_lang": self.src_lang, "nodes": len(self.nodes),
                "titles": {k: v["text"][:120] for k, v in self._titles.items()},
                "articles": len(self.articles), "ambiguous_articles": len(self.ambiguous_articles),
                "definitions": len(self.definitions), "definition_terms": len(self._defs_by_key),
                "definitions_by_source": dict(kinds),
                "quoted_nodes": sum(1 for v in self._quoted.values() if v),
                "quote_scopes": len(self._quote_chain)}

    def check_doc(self, doc_id, unit):
        if self.doc_id is not None and doc_id != self.doc_id and getattr(unit, "doc_id", None) != self.doc_id:
            raise ValueError(f"SourceIndex of {self.doc_id!r} used for {doc_id!r}: cross-document context")

    def title_item(self, scope_id, spans) -> Optional[EvidenceItem]:
        top = self.top_of(scope_id)
        t = self._titles.get(top) or self._titles.get("main")
        if not t or not t["text"]:
            return None
        if _overlaps(t["spans"], spans):
            return None
        scope = top if top in self._titles else "main"
        return EvidenceItem("title_path", "source_title", t["text"], 0,
                            ref={"node_ids": list(t["node_ids"]), "scope": scope})

    def path_item(self, unit_nodes, spans) -> Optional[EvidenceItem]:
        if not unit_nodes:
            return None
        first = unit_nodes[0]
        scope = first.scope_id or "main"
        if _is_quote_scope(scope):
            chain = list(self._quote_chain.get(scope, []))
            inner = None
            for h in self._by_label_scope(scope):
                if h.kind in HEADING_KINDS and h.spans[0][0] <= first.raw_start:
                    if inner is None or h.spans[0][0] > inner.spans[0][0]:
                        inner = h
            if inner is not None:
                chain.append(inner)
        else:
            cp = _canon_path(first)
            chain = self._resolve_chain(scope, cp)
        comps = [h for h in chain if not _overlaps(h.spans, spans)]
        if not comps:
            return None
        return EvidenceItem("title_path", "source_path", " > ".join(h.text for h in comps), 0,
                            ref={"node_ids": [h.node_id for h in comps], "scope": scope})

    def _by_label_scope(self, scope):
        for (sc, _lab), hs in self._by_label.items():
            if sc == scope:
                yield from hs

    def definition_items(self, text, scope_id, spans, max_items) -> list:
        if max_items <= 0 or not text or not self._defs_by_key:
            return []
        tf = _fold(text)
        top = self.top_of(scope_id)
        found = []
        for key, defs in self._defs_by_key.items():
            if key not in tf:
                continue
            ms = [(m.start(), m.end()) for m in self._rx(key, defs[0].term).finditer(text)]
            if ms:
                found.append((key, ms, defs))
        found.sort(key=lambda x: (-len(x[0]), x[1][0][0]))
        consumed, chosen = [], []
        for key, ms, defs in found:
            free = [s for s in ms if not _overlaps([s], consumed)]
            if not free:
                continue
            consumed.extend(free)
            usable = [d for d in defs if not _overlaps([d.span], spans)]
            if not usable:
                continue
            ref_pos = free[0][0]
            usable.sort(key=lambda d: (d.top != top, d.source != "definition_article", d.pos))
            chosen.append((usable[0], ref_pos))
        chosen.sort(key=lambda x: (x[0].top != top, -len(x[0].key), x[1]))
        out, seen_text = [], set()
        for d, _p in chosen:
            if d.text in seen_text:
                continue
            seen_text.add(d.text)
            out.append(EvidenceItem("definition", "source_definition", d.text, 0,
                                    ref={"term": d.term, "node_id": d.node_id, "source": d.source,
                                         "scope": d.scope}))
            if len(out) >= max_items:
                break
        return out

    def reference_items(self, text, starts, unit_nodes, scope_id, spans, max_items) -> list:
        """Article N / Điều N mentioned in the unit -> heading of Article N of THIS document (same
        top scope), unless qualified by another document, in an amendment instruction, quoted,
        ambiguous, missing here, or the unit's own article."""
        if max_items <= 0 or not text or _is_quote_scope(scope_id):
            return []
        lang = self.src_lang
        top = self.top_of(scope_id)
        own = set()
        for n in unit_nodes:
            for lab in _canon_path(n) + ([_own_label(n)] if _own_label(n) else []):
                if lab and lab[0] == "article":
                    own.add(lab[1])
        aligned = starts is not None and len(starts) == len(unit_nodes) and len(unit_nodes) > 0
        out, seen = [], set()
        for m in _REF[lang].finditer(text):
            if aligned:
                k = max(0, bisect_right(starts, m.start()) - 1)
                if self._quoted_at(unit_nodes[k], m.start() - starts[k]):
                    continue
            # amendment instructions ("Article N is amended as follows", "sửa đổi Điều N") point to
            # the amended document, not this one: windows stay inside the sentence and the node
            before = text[max(0, m.start() - 120):m.start()]
            before = before[before.rfind("\n") + 1:]
            before = before[_sentence_start(before, len(before)):]
            after = text[m.end():m.end() + 120].split("\n", 1)[0]
            after_sent = re.split(r"(?<=[.!?])\s+(?=[\"“(A-ZĐ])|;", after, maxsplit=1)[0]
            if _AMEND_BEFORE[lang].search(before) or _AMEND_AFTER[lang].search(after_sent):
                continue
            kind, doc_word = _qualifier(after, lang)
            if kind == "external":
                continue
            target_top = top
            if doc_word and top != "main" and "main" in self._titles:
                stem = doc_word.rstrip("s") if lang == "en" else doc_word
                mine = self._title_doc_word.get(top, "")
                main = self._title_doc_word.get("main", "")
                if stem and stem in main and stem not in mine:
                    target_top = "main"
            nums = [_canon_num(x) for x in _NUM.findall(m.group(1) + " " + (m.group(2) or ""))]
            for num in nums:
                if target_top == top and num in own:
                    continue
                key = (target_top, num)
                if key in seen or key in self.ambiguous_articles:
                    continue
                h = self.articles.get(key)
                if h is None or _overlaps(h.spans, spans):
                    continue
                seen.add(key)
                out.append(EvidenceItem("reference", "source_reference", h.text, 0,
                                        ref={"article": num, "node_id": h.node_id, "scope": h.scope}))
                if len(out) >= max_items:
                    return out
        return out

    def _quoted_at(self, node, local_pos) -> bool:
        if _is_quote_scope(node.scope_id):
            return True
        payload = node.payload or ""
        d = self._depth.get(node.node_id, 0)
        d += payload.count("“", 0, local_pos) - payload.count("”", 0, local_pos)
        if d > 0:
            return True
        rest = payload[local_pos:]
        o, c = rest.find("“"), rest.find("”")
        return c >= 0 and (o < 0 or c < o)

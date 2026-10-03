"""Structure parser (agent P): raw SOURCE text -> list[Node] that partition the source exactly.

Contract: SPEC.md section 3 "parse.py" (draft section 2 wins on conflict). Only the source side of a
doc-direction is ever read here; nothing in this module looks at a reference/gold file.

Pipeline (line based, offsets kept on the raw string, raw text never modified):
  1. lines + features (indent, cells split by >=3 spaces, upper-case ratio, labels)
  2. header regions (national motto anchored), table regions (numeric anchors / pipe rows)
  3. text blocks: blank lines end a block; inside a run of lines a new block starts at a structure
     label or a short/sentence-final line (PDF text often has no blank line between paragraphs)
  4. page-break join across blank lines (SPEC rule + connector-word extension, flagged)
  5. signature blocks, recipients, attachment/appendix scopes, quote scopes, paths, protected
  6. content spans -> Nodes; whitespace between spans goes to trail (up to the last newline) and
     lead (indentation / column gap). Invariant: "".join(n.raw) == src.
"""
# Split from lam_docmt/parse.py (statements verbatim; tools/split_from_lam_docmt.py).
from __future__ import annotations

import re
from dataclasses import dataclass, field

try:  # keep the table cap in sync with the run config default
    from ..core.config import RunConfig

    TABLE_CAP = RunConfig().table_cap
except Exception:  # pragma: no cover
    TABLE_CAP = 384

# ----------------------------------------------------------------------------------------------
# regexes and word lists (generic, no document-specific values)
# ----------------------------------------------------------------------------------------------
TERMINAL = set('.;:!?)”"')          # SPEC page-break rule: paragraph "ends" with one of these

SENT_END = set(".;:!?")

HEADING_KINDS = ("article", "chapter", "section", "part")

BODY_KINDS = ("paragraph", "clause", "point", "list_item", "article", "unknown")

_ART_RE = re.compile(
    r"(Article|ARTICLE|Điều|ĐIỀU)[ \t]+(\d{1,4}[a-zđ]?|\d{1,4} [a-zđ](?=\.))(?!\d)[ \t]*"
    r"((?:[.:][ \t]?[-–~])|[.:\-–~])?")

_CHAP_RE = re.compile(r"(Chapter|CHAPTER|Chương|CHƯƠNG)[ \t]+([IVXLCDM]{1,7}|\d{1,3})(?![\w])[ \t]*([.:\-–]+)?")

_SECT_RE = re.compile(r"(Section|SECTION|Mục|MỤC)[ \t]+([IVXLCDM]{1,7}|\d{1,3})(?![\w])[ \t]*([.:\-–]+)?")

_PART_RE = re.compile(
    r"(Part|PART|Phần|PHẦN)[ \t]+([IVXLCDM]{1,7}|\d{1,3}|ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|"
    r"One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|THỨ [A-ZÀ-ỸĐ]+)(?![\w])[ \t]*([.:\-–]+)?")

_ROMAN_RE = re.compile(r"([IVXLC]{1,6})[ \t]?(\.[ \t]?-|\.|-|–)(?=[ \t]+\S)")

_ULET_RE = re.compile(r"([A-ZĐ])[ \t]?(\.|\))(?=[ \t]+\S)")

_CLAUSE_RE = re.compile(
    r"(?:(\d{1,3}(?:\.\d{1,3}){0,3}\.(?:[ \t]?-)?)(?=[ \t]|$|[^\d\s])"
    r"|(\d{1,3}\))(?=[ \t]|$|[^\d\s])"
    r"|(\d{1,3}[ \t]?/)(?=[ \t])"
    r"|(\d{1,3}[ \t]?[-–])(?=[ \t]*[^\d\s]))")

_POINT_RE = re.compile(r"((?:[a-zđ]|dd|đđ)[ \t]?[)/](?=[ \t]|$|[^\s\d])|[a-zđ]\.(?=[ \t]))")

_BULLET_RE = re.compile(r"([-+•*–—∙·◦▪●○■□➢✓])(?=[ \t]+\S)")

_QUOTE_PREFIX_RE = re.compile(r"^[“\"‘«„]\s*")

_MOTTO_RE = re.compile(r"Independence\s*[-–]\s*Freedom|INDEPENDENCE\s*[-–]\s*FREEDOM|"
                       r"Độc\s+[Ll]ập\s*[-–]\s*Tự\s+[Dd]o|ĐỘC\s+LẬP\s*[-–]\s*TỰ\s+DO")

_REPUBLIC_RE = re.compile(r"SOCIALIST REPUBLIC OF VIET|CỘNG HO[ÀÒ] XÃ HỘI CHỦ NGHĨA")

_HDR_NUMBER_RE = re.compile(r"^(No|NO|Số|SỐ|Ref)\s*[.:]")

_HDR_DATE_RE = re.compile(r"(Ha\s?noi|Hà Nội|HÀ NỘI|Ho Chi Minh City|TP\.)\s*,|ngày[\s.…]*\d*[\s.…]*tháng|"
                          r"^[.…\s]*,\s*(ngày|date|dated)", re.I)

_PROMUL_NOTE_RE = re.compile(r"^\(?\s*(Promulgated|Issued|Enclosed|Attached)\s+(together\s+)?with\b|"
                             r"^\(?\s*(Ban hành\s+)?[Kk]èm\s+theo\b|^\(?\s*Ban hành kèm\b", re.I)

_APPENDIX_RE = re.compile(r"^(PHỤ LỤC|Phụ lục|APPENDIX|Appendix|ANNEX|Annex|Mẫu số|MẪU SỐ|Form No\.?|FORM NO\.?|"
                          r"Table\s+\d+|TABLE\s+\d+|Bảng\s+\d+|BẢNG\s+\d+|Biểu mẫu|BIỂU MẪU)(?![\w])")

_SIG_TITLE_RE = re.compile(
    r"(?:\b(?:PRIME MINISTER|MINISTER|CHAIRMAN|CHAIRWOMAN|CHAIRPERSON|PRESIDENT|DEPUTY|DIRECTOR|"
    r"CHIEF JUSTICE|CHIEF JUDGE|INSPECTOR GENERAL|GOVERNOR|SECRETARY|ACTING|FOR THE|ON BEHALF)\b"
    r"|\bPP\b\.?|\bP\.P\.|BỘ TRƯỞNG|THỨ TRƯỞNG|THỦ TƯỚNG|CHỦ TỊCH|CHÁNH ÁN|VIỆN TRƯỞNG|TỔNG THANH TRA|"
    r"GIÁM ĐỐC|CỤC TRƯỞNG|CHỦ NHIỆM|THAY MẶT|\bKT\.|\bTM\.|\bTL\.|\bTUQ\.|ĐẠI TƯỚNG|THƯỢNG TƯỚNG)")

_SIGNED_RE = re.compile(r"^\(\s*(signed|đã ký|sealed|ký|signed and sealed)\b[^)]*\)\.?$", re.I)

_DATE_LINE_RE = re.compile(r"^(Date|Ngày|Dated)\s*:", re.I)

_RECIP_RE = re.compile(r"^(Nơi nhận|NƠI NHẬN|Recipients?|RECIPIENTS?|Nơi gửi)\b")

_NUMCELL_RE = re.compile(r"\(?[+\-]?\d+(?:[.,]\d+)*\)?(?:[ \t]*[-–/x×][ \t]*\(?\d+(?:[.,]\d+)*\)?)*[ \t]*%?\.?")

_PIPE_SEP_RE = re.compile(r"^\|?(\s*:?-{2,}:?\s*\|)+\s*(:?-{2,}:?\s*)?\|?$")

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")

_CONNECT_EN = {"of", "the", "a", "an", "to", "in", "on", "at", "by", "for", "with", "from", "under", "between",
               "into", "upon", "as", "that", "which", "its", "their", "his", "her", "this", "these", "those",
               "such", "per", "than", "via", "within", "without", "against", "over", "about", "after",
               "before", "during", "including", "regarding", "concerning", "pursuant", "according", "whose",
               "who", "whom", "is", "are", "was", "were", "be", "been", "being", "shall", "may", "must",
               "will", "would", "should", "can", "could", "not", "has", "have", "had", "if", "when", "where",
               "whether", "because", "while", "through", "toward", "towards", "among", "amongst", "out",
               "up", "down", "off", "onto", "unto", "via", "both", "either", "neither", "each", "every",
               "all", "any", "some", "no", "other", "another", "same", "said", "very", "also"}

_CONNECT_VI = {"của", "các", "những", "theo", "tại", "về", "với", "cho", "trong", "đến", "từ", "là", "được",
               "bị", "có", "do", "bởi", "khi", "nếu", "mà", "thì", "để", "như", "đối", "giữa", "trên", "dưới",
               "sau", "trước", "qua", "vào", "ra", "một", "mọi", "số", "ngày", "tháng", "năm", "và", "hoặc",
               "hay", "nhưng", "nên", "vì", "tới", "phải", "sẽ", "đã", "đang", "chưa", "không", "cũng",
               "rằng", "nhằm", "việc", "sự", "cùng", "kể", "gồm", "bao", "thuộc", "trực", "này", "đó"}

_CONNECT_WEAK = {"and", "or", "nor", "và", "hoặc", "hay"}

_REF_NOUNS = {"clause", "clauses", "article", "articles", "point", "points", "item", "items", "chapter",
              "chapters", "section", "sections", "part", "parts", "annex", "appendix", "form", "table", "no",
              "nos", "khoản", "điểm", "điều", "chương", "mục", "phần", "phụ", "lục", "mẫu", "bảng", "số",
              "decree", "circular", "decision", "law", "resolution", "ordinance", "code", "directive",
              "luật", "nghị", "thông", "quyết", "pháp", "lệnh", "bộ"}

_ABBREV_TAIL_RE = re.compile(r"(?:\b(?:No|Nos|NO|Art|Arts|Mr|Mrs|Ms|Dr|Prof|St|Co|Ltd|Inc|vs|Cl|Para|para|pp|"
                             r"Vol|vol|TT|QĐ|Đ|đ|Tp|TP|e\.g|i\.e)\.|\b(?:Số|số)\s*[.:]?)$")

_NAME_FIRST_STOP = {"the", "for", "on", "of", "and", "to", "in", "at", "by", "date", "ngày", "nơi", "kính",
                    "căn", "theo", "pursuant", "dear", "mr", "mrs", "ms", "ông", "bà", "this", "that", "these",
                    "article", "chapter", "section", "điều", "chương", "mục", "phần", "table", "bảng"}

_NAME_ANY_STOP = {"minister", "ministry", "government", "president", "chairman", "chairwoman", "committee",
                  "office", "department", "council", "assembly", "court", "procuracy", "republic", "vietnam",
                  "viet", "nam", "hanoi", "independence", "freedom", "happiness", "deputy", "director",
                  "general", "chief", "inspector", "secretary", "head", "people", "people's", "people’s",
                  "national", "state", "law", "decree", "circular", "decision", "signed", "province", "city",
                  "district", "agency", "bureau", "of", "and", "for", "the", "on", "in", "by", "to", "with",
                  "chính", "phủ", "bộ", "trưởng", "thủ", "tướng", "chủ", "tịch", "quốc", "hội", "ủy", "ban",
                  "nhân", "dân", "tòa", "án", "viện", "kiểm", "sát", "cộng", "hòa", "hoà", "xã", "nghĩa",
                  "việt", "độc", "lập", "tự", "do", "hạnh", "phúc", "nơi", "nhận", "và", "của", "các",
                  "thay", "mặt", "giám", "đốc", "thứ", "phó", "chánh", "cục", "vụ", "sở", "đã", "ký"}

_RANK_FIRST = {"general", "lieutenant", "colonel", "major", "senior", "captain", "admiral", "marshal",
               "brigadier"}

_RANK_PAIR = {"thượng tướng", "đại tướng", "trung tướng", "thiếu tướng", "đại tá", "thượng tá", "trung tá",
              "thiếu tá"}

_NAME_TOKEN_RE = re.compile(r"^[^\W\d_](?:[^\W\d_]|['’\-](?=[^\W\d_]))*$")

def _has_letter(s: str) -> bool:
    return any(c.isalpha() for c in s)

def _has_digit(s: str) -> bool:
    return any(c.isdigit() for c in s)

def _upper_ratio(s: str):
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 4:
        return None
    return sum(c.isupper() for c in letters) / len(letters)

def _last_word(s: str) -> str:
    ws = _WORD_RE.findall(s)
    return ws[-1].lower() if ws else ""

def _last_two(s: str) -> str:
    ws = _WORD_RE.findall(s)
    return " ".join(w.lower() for w in ws[-2:]) if len(ws) >= 2 else ""

def _ends_abbrev(s: str) -> bool:
    return bool(_ABBREV_TAIL_RE.search(s.rstrip()))

def _strong_cont(s: str) -> bool:
    """The text visibly continues on the next line (comma, open bracket, hyphenation, connector)."""
    s = s.rstrip()
    if not s:
        return False
    if s[-1] in ",(/[{“‘" or (s[-1] == "-" and len(s) > 1 and s[-2].isalpha()):
        return True
    if s.count("(") > s.count(")"):
        return True
    if _ends_abbrev(s):
        return True
    if not s[-1].isalpha():
        return False
    w = _last_word(s)
    return w in _CONNECT_EN or w in _CONNECT_VI or w in _REF_NOUNS

def _starts_upperish(s: str) -> bool:
    c = s[:1]
    return bool(c) and (c.isupper() or c.isdigit() or c in "“\"‘(«[")

# ----------------------------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------------------------
@dataclass
class _Label:
    kind: str           # article | chapter | section | part | clause | point | list_item
    level: str          # part | chapter | section | article | item | clause | subclause | point | bullet
    label: str          # label text as in the source ("1.", "a)", "Article 6.-")
    path_label: str     # label used in Node.path ("Article 6", "Chương II", "1.", "a)")
    number: str = ""    # heading number (protected)
    rest: str = ""      # text after the label on the same line
    prefix: str = ""    # opening quote before the label, if any

def _kw_norm(kw: str) -> str:
    k = kw.lower()
    return {"article": "Article", "điều": "Điều", "chapter": "Chapter", "chương": "Chương", "section": "Section",
            "mục": "Mục", "part": "Part", "phần": "Phần"}.get(k, kw)

def match_label(s: str) -> Optional[_Label]:
    """Structure label at the start of a stripped line (an opening quote is allowed before it)."""
    if not s:
        return None
    m0 = _QUOTE_PREFIX_RE.match(s)
    prefix, t = (m0.group(0), s[m0.end():]) if m0 else ("", s)
    if not t:
        return None
    m = _ART_RE.match(t)
    if m:
        rest = t[m.end():]
        if (m.group(3) and not rest[:1].isdigit()) or not rest.strip():
            num = m.group(2)
            kw = _kw_norm(m.group(1))
            return _Label("article", "article", t[:m.end()].strip(), f"{kw} {num}", num, rest.strip(), prefix)
    for rx, kind in ((_CHAP_RE, "chapter"), (_SECT_RE, "section"), (_PART_RE, "part")):
        m = rx.match(t)
        if m:
            rest = t[m.end():]
            ur = _upper_ratio(rest)
            if m.group(3) or not rest.strip() or (ur is not None and ur >= 0.8):
                num = m.group(2)
                kw = _kw_norm(m.group(1))
                return _Label(kind, kind, t[:m.end()].strip(), f"{kw} {num}", num, rest.strip(), prefix)
    m = _ROMAN_RE.match(t)
    if m and (len(m.group(1)) > 1 or m.group(1) in "IVX") and not t[m.end():].strip()[:1].isdigit():
        lab = t[:m.end()].strip()
        return _Label("section", "section", lab, lab, m.group(1), t[m.end():].strip(), prefix)
    m = _ULET_RE.match(t)
    if m and m.group(1) not in "IVX":
        lab = t[:m.end()].strip()
        return _Label("clause", "item", lab, lab, "", t[m.end():].strip(), prefix)
    m = _CLAUSE_RE.match(t)
    if m:
        lab = t[:m.end()].strip()
        nums = re.findall(r"\d+", lab)
        level = "subclause" if len(nums) >= 2 else "clause"
        return _Label("clause", level, lab, lab, "", t[m.end():].strip(), prefix)
    m = _POINT_RE.match(t)
    if m:
        lab = t[:m.end()].strip()
        return _Label("point", "point", lab, lab, "", t[m.end():].strip(), prefix)
    m = _BULLET_RE.match(t)
    if m:
        return _Label("list_item", "bullet", m.group(1), m.group(1), "", t[m.end():].strip(), prefix)
    return None

# ----------------------------------------------------------------------------------------------
# signer names
# ----------------------------------------------------------------------------------------------
def name_score(text: str) -> float:
    """Confidence that a stripped line/cell is a signer's personal name (0 = not a name).
    >= 0.8 only for 2-5 Title-Case letter-only words, no title/institution word, no digits."""
    t = text.strip()
    core = t.rstrip("*").rstrip()
    suffix = core != t
    toks = core.split()
    if not 2 <= len(toks) <= 6 or _has_digit(core):
        return 0.0
    if not all(_NAME_TOKEN_RE.match(x) for x in toks):
        return 0.0
    low = [x.lower() for x in toks]
    if low[0] in _NAME_FIRST_STOP:
        return 0.0
    rank = low[0] in _RANK_FIRST or " ".join(low[:2]) in _RANK_PAIR
    core_low = low[2:] if " ".join(low[:2]) in _RANK_PAIR else (low[1:] if low[0] in _RANK_FIRST else low)
    if any(x in _NAME_ANY_STOP for x in core_low):
        return 0.0
    if all(x.isupper() for x in toks):
        return 0.5
    caps = [x[0].isupper() for x in toks]
    if rank:
        return 0.6 if len(toks) <= 6 else 0.0
    if len(toks) > 5:
        return 0.0
    if all(caps) and all(x[1:] == x[1:].lower() for x in toks):
        score = 0.9 if len(toks) <= 4 else 0.85
    elif caps[0] and sum(caps) >= 2:
        score = 0.6          # e.g. a lower-case syllable: evidence too weak to protect
    else:
        return 0.0
    if suffix:
        score = 0.8
    return score

# ----------------------------------------------------------------------------------------------
# internal structures
# ----------------------------------------------------------------------------------------------
@dataclass
class _Line:
    i: int
    start: int
    end: int
    text: str
    s: str
    indent: int
    blank: bool
    cstart: int = 0       # absolute offset of the first non-space char
    cend: int = 0         # absolute offset after the last non-space char
    cells: list = field(default_factory=list)   # [(abs_s, abs_e)] split by >= 3 spaces

@dataclass
class _Span:
    s: int
    e: int
    kind: str
    translatable: bool = True
    confidence: float = 1.0
    protected: dict = field(default_factory=dict)
    table: Optional[dict] = None
    flags: list = field(default_factory=list)
    scope_id: str = "main"
    path: list = field(default_factory=list)

@dataclass
class _Block:
    a: int                      # first line index
    b: int                      # last line index + 1
    typ: str = "text"           # text | columnar | header | table | sig
    kind: str = "paragraph"
    label: Optional[_Label] = None
    conf: float = 1.0
    flags: list = field(default_factory=list)
    scope_id: str = "main"
    spans: list = field(default_factory=list)   # filled for header/table/sig/columnar blocks
    table_id: Optional[str] = None

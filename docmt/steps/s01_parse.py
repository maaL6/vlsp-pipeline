from __future__ import annotations

import math
import re
from collections import Counter

from ..core.datatypes import NODE_KINDS, Node
from .s01_parse_lines import BODY_KINDS, HEADING_KINDS, SENT_END, TABLE_CAP, TERMINAL, _APPENDIX_RE, _BULLET_RE, _Block, _CONNECT_WEAK, _DATE_LINE_RE, _HDR_DATE_RE, _HDR_NUMBER_RE, _Line, _MOTTO_RE, _NUMCELL_RE, _PIPE_SEP_RE, _PROMUL_NOTE_RE, _RECIP_RE, _REPUBLIC_RE, _SIGNED_RE, _SIG_TITLE_RE, _Span, _ends_abbrev, _has_digit, _has_letter, _last_word, _starts_upperish, _strong_cont, _upper_ratio, match_label, name_score
from .s02_payload import make_payload, unwrap

PARSER_VERSION = "p1.0"

class _Parser:
    def __init__(self, src: str, doc_id: str, src_lang: str, tok=None):
        self.src = src
        self.doc_id = doc_id
        self.src_lang = src_lang
        self.tok = tok
        self.lines: list[_Line] = []
        pos = 0
        for i, raw in enumerate(src.split("\n")):
            start, end = pos, pos + len(raw)
            pos = end + 1
            st = raw.strip()
            ind = len(raw) - len(raw.lstrip(" \t"))
            ln = _Line(i, start, end, raw, st, ind, not st)
            if st:
                lead_ws = len(raw) - len(raw.lstrip())
                trail_ws = len(raw) - len(raw.rstrip())
                ln.cstart = start + lead_ws
                ln.cend = end - trail_ws
                ln.cells = self._cells(ln, 3)
            self.lines.append(ln)
        self.n = len(self.lines)
        body = [ln for ln in self.lines if not ln.blank and len(ln.cells) < 2]
        ind = Counter(ln.indent for ln in body)
        self.base_indent = ind.most_common(1)[0][0] if ind else 0
        lens = sorted(len(ln.text.rstrip()) for ln in body if ln.indent <= self.base_indent + 4)
        if len(lens) >= 20:
            self.W = max(40, lens[int(0.98 * (len(lens) - 1))])
        else:
            self.W = max([80] + lens)
        self._label_cache: dict = {}
        self.header_lines: set = set()
        self.table_lines: dict = {}      # line index -> table region id
        self.tables: list = []           # [(a, b, kind)] kind: plain | pipe

    # -- line helpers -----------------------------------------------------------------------
    def _cells(self, ln: _Line, gap: int):
        if ln.blank:
            return []
        txt = ln.text
        rx = re.compile(r"\S+(?: {1,%d}\S+)*" % (gap - 1))
        return [(ln.start + m.start(), ln.start + m.end()) for m in rx.finditer(txt)]

    def label(self, ln: _Line) -> Optional[_Label]:
        if ln.i not in self._label_cache:
            self._label_cache[ln.i] = match_label(ln.s) if not ln.blank else None
        return self._label_cache[ln.i]

    def upper(self, ln: _Line) -> bool:
        r = _upper_ratio(ln.s)
        return r is not None and r >= 0.8

    def centered(self, ln: _Line) -> bool:
        return ln.indent >= max(8, self.base_indent + 6)

    def text(self, s: int, e: int) -> str:
        return self.src[s:e]

    # -- 1. header regions ------------------------------------------------------------------
    def _hdr_like(self, ln: _Line) -> bool:
        if ln.blank:
            return False
        s = ln.s
        if len(ln.cells) >= 2:
            return True
        if not _has_letter(s) and not _has_digit(s):
            return True
        return bool(_MOTTO_RE.search(s) or _REPUBLIC_RE.search(s) or _HDR_NUMBER_RE.match(s)
                    or _HDR_DATE_RE.search(s))

    def find_headers(self):
        for ln in self.lines:
            if ln.blank or not _MOTTO_RE.search(ln.s) or ln.i in self.header_lines:
                continue
            region = {ln.i}
            for step in (-1, 1):
                k, blanks, taken = ln.i + step, 0, 0
                limit = 3 if step < 0 else 4
                while 0 <= k < self.n and taken < limit:
                    x = self.lines[k]
                    if x.blank:
                        blanks += 1
                        if blanks > 1:
                            break
                        k += step
                        continue
                    if not self._hdr_like(x) or match_label(x.s) and match_label(x.s).kind in HEADING_KINDS:
                        break
                    if step > 0 and len(x.cells) < 2 and _has_letter(x.s) and not (
                            _MOTTO_RE.search(x.s) or _REPUBLIC_RE.search(x.s) or _HDR_NUMBER_RE.match(x.s)
                            or _HDR_DATE_RE.search(x.s)):
                        break
                    region.add(k)
                    blanks = 0
                    taken += 1
                    k += step
            self.header_lines |= region

    # -- 2. tables -------------------------------------------------------------------------
    def _is_note(self, s: str) -> bool:
        return bool(_PROMUL_NOTE_RE.match(s))

    def _numeric_cell(self, s: str) -> bool:
        return bool(_NUMCELL_RE.fullmatch(s.strip()))

    def _anchor(self, ln: _Line):
        """'strong' = columnar line with a numeric cell, 'weak' = numeric-only line, None otherwise."""
        if ln.blank or ln.i in self.header_lines:
            return None
        if len(ln.cells) >= 2:
            c2 = [self.text(a, b) for a, b in self._cells(ln, 2)]
            if any(self._numeric_cell(c) for c in c2):
                return "strong"
            return None
        if self._numeric_cell(ln.s) and len(ln.s) <= 20:
            return "weak"
        return None

    def _table_like(self, ln: _Line, tind: int) -> bool:
        if ln.blank or ln.i in self.header_lines or ln.i in self.table_lines:
            return False
        s = ln.s
        if self._is_note(s) or _APPENDIX_RE.match(s):
            return False
        lab = self.label(ln)
        if lab and lab.kind in HEADING_KINDS:
            return False
        if ln.indent < tind:
            return False
        if len(self._cells(ln, 2)) >= 2:
            return True
        if self._numeric_cell(s):
            return True
        if lab and lab.kind in ("clause", "point", "list_item") and ln.indent <= self.base_indent \
                and tind <= self.base_indent:
            return False
        return len(s) <= 0.75 * self.W or ln.indent >= tind + 6

    def find_tables(self):
        # pipe tables first
        k = 0
        while k < self.n:
            ln = self.lines[k]
            if not ln.blank and ln.s.startswith("|") and ln.s.count("|") >= 2 and k not in self.header_lines:
                j = k
                while j < self.n and not self.lines[j].blank and self.lines[j].s.startswith("|") \
                        and self.lines[j].s.count("|") >= 2:
                    j += 1
                if j - k >= 2:
                    tid = f"T{len(self.tables) + 1}"
                    self.tables.append((k, j, "pipe", tid))
                    for x in range(k, j):
                        self.table_lines[x] = tid
                k = j
            else:
                k += 1
        anchors = [(ln.i, self._anchor(ln)) for ln in self.lines if ln.i not in self.table_lines]
        anchors = [(i, a) for i, a in anchors if a]
        clusters, cur = [], []
        for i, a in anchors:
            if cur and i - cur[-1][0] > 8:
                clusters.append(cur)
                cur = []
            cur.append((i, a))
        if cur:
            clusters.append(cur)
        for cl in clusters:
            strong = [i for i, a in cl if a == "strong"]
            if len(cl) < 2 or not strong:
                continue
            fa, la = cl[0][0], cl[-1][0]
            if any(self.lines[x].i in self.table_lines for x in range(fa, la + 1)):
                continue
            tind = min(self.lines[i].indent for i in strong)
            # lines between anchors that are hard stops split the cluster: keep it simple, stop on headings
            a = fa
            k, blanks = fa - 1, 0
            while k >= 0:
                x = self.lines[k]
                if x.blank:
                    blanks += 1
                    if blanks > 1:
                        break
                    k -= 1
                    continue
                if not self._table_like(x, tind):
                    break
                blanks = 0
                a = k
                k -= 1
            b = la + 1
            k, blanks = la + 1, 0
            while k < self.n:
                x = self.lines[k]
                if x.blank:
                    blanks += 1
                    if blanks > 1:
                        break
                    k += 1
                    continue
                if not self._table_like(x, tind) and not self._anchor(x):
                    break
                blanks = 0
                b = k + 1
                k += 1
            tid = f"T{len(self.tables) + 1}"
            self.tables.append((a, b, "plain", tid))
            for x in range(a, b):
                self.table_lines[x] = tid
        self.tables.sort()

    # -- 3. segmentation --------------------------------------------------------------------
    def split_between(self, p: _Line, c: _Line) -> bool:
        ps, cs = p.s, c.s
        clab, plab = self.label(c), self.label(p)
        if cs[0].islower() and not (clab and clab.kind == "point"):
            return False
        p_up, c_up = self.upper(p), self.upper(c)
        if p_up and c_up:
            if clab and clab.kind in HEADING_KINDS:
                return True
            if plab and plab.kind in ("chapter", "section", "part") and not plab.rest:
                return True
            if _strong_cont(ps) or _last_word(ps) in _CONNECT_WEAK:
                return False
            if ps[-1] in ".:;!?":
                return True
            return len(ps) < 0.5 * self.W
        if p_up != c_up:
            if _strong_cont(ps) and not clab:
                return False
            return True
        if self.centered(p) != self.centered(c):
            if _strong_cont(ps):
                return False
            return True
        pcont = _strong_cont(ps)
        first = cs.split()[0]
        fill = (len(p.text.rstrip()) + 1 + len(first)) / max(self.W, 1)
        if clab:
            return not pcont
        if ps[-1] in SENT_END and not _ends_abbrev(ps) and fill < 0.85 and _starts_upperish(cs):
            return True
        if fill < 0.6 and _starts_upperish(cs) and not pcont:
            return True
        return False

    def segment(self) -> list:
        blocks: list[_Block] = []
        cur: Optional[_Block] = None
        table_done = set()
        for ln in self.lines:
            i = ln.i
            if i in self.table_lines:
                cur = None
                tid = self.table_lines[i]
                if tid not in table_done:
                    table_done.add(tid)
                    a, b, kind, _ = next(t for t in self.tables if t[3] == tid)
                    blocks.append(_Block(a, b, typ="table", kind="table_cell", table_id=tid))
                continue
            if ln.blank:
                cur = None
                continue
            if i in self.header_lines:
                cur = None
                blocks.append(_Block(i, i + 1, typ="header", kind="header_cell"))
                continue
            if len(ln.cells) >= 2:
                cur = None
                blocks.append(_Block(i, i + 1, typ="columnar", kind="unknown"))
                continue
            if cur is not None and not self.split_between(self.lines[cur.b - 1], ln):
                cur.b = i + 1
                continue
            cur = _Block(i, i + 1)
            blocks.append(cur)
        return blocks

    # -- 4. classification -----------------------------------------------------------------
    def block_text(self, bl: _Block) -> str:
        return self.src[self.lines[bl.a].cstart:self.lines[bl.b - 1].cend]

    def classify(self, bl: _Block):
        if bl.typ != "text":
            return
        first = self.lines[bl.a]
        txt = self.block_text(bl)
        lab = self.label(first)
        bl.label = None
        if not _has_letter(txt):
            bl.kind = "separator"
            return
        if lab:
            bl.kind = lab.kind
            bl.label = lab
            return
        if _MOTTO_RE.search(first.s) or _REPUBLIC_RE.search(first.s) and len(first.s) < 60:
            bl.kind = "header_cell"
            return
        st = first.s
        if (st.startswith("(") and (self._is_note(st) or self.centered(first))) or self._is_note(st):
            bl.kind = "note"
            return
        ur = _upper_ratio(unwrap(txt))
        if ur is not None and ur >= 0.8:
            bl.kind = "title"
            return
        bl.kind = "paragraph"

    # -- 5. page-break join -----------------------------------------------------------------
    def join_page_breaks(self, blocks: list) -> list:
        out: list[_Block] = []
        for bl in blocks:
            prev = out[-1] if out else None
            if prev is not None and prev.typ == "text" and bl.typ == "text" and self._page_join(prev, bl):
                upper_start = not self.lines[bl.a].s[0].islower()
                prev.b = bl.b
                if "page_break_join" not in prev.flags:
                    prev.flags.append("page_break_join")
                if upper_start:
                    prev.conf = min(prev.conf, 0.8)
                    if "page_break_join_upper" not in prev.flags:
                        prev.flags.append("page_break_join_upper")
                continue
            out.append(bl)
        return out

    def _page_join(self, prev: _Block, nxt: _Block) -> bool:
        if prev.kind not in BODY_KINDS or nxt.kind not in ("paragraph", "unknown"):
            return False
        gap = nxt.a - prev.b          # blank lines in between
        if gap < 1 or gap > 2:
            return False
        last = self.lines[prev.b - 1]
        first = self.lines[nxt.a]
        if last.s[-1] in TERMINAL:
            return False
        if self.centered(first) and not self.centered(last):
            return False
        if first.s[0].islower():
            return True
        # extension: upper-case start (proper nouns, "No. 26/...") only after a connector word
        lw = last.s
        if (_strong_cont(lw) or _last_word(lw) in _CONNECT_WEAK) and not self.upper(first):
            return True
        return False

    # -- 6. signature blocks / scopes -------------------------------------------------------
    def _title_followed_by_note(self, k: int) -> bool:
        ln = self.lines[k]
        if ln.blank or not self.upper(ln):
            return False
        seen = 0
        j = k + 1
        while j < self.n and seen < 3:
            x = self.lines[j]
            if not x.blank:
                seen += 1
                if self._is_note(x.s):
                    return True
            j += 1
        return False

    def _cell_texts(self, ln: _Line):
        return [self.text(a, b) for a, b in ln.cells] if len(ln.cells) >= 2 else [ln.s]

    def _sig_compatible(self, ln: _Line, name_seen: bool) -> bool:
        cells = self._cell_texts(ln)
        if any(_BULLET_RE.match(c) or _RECIP_RE.match(c) for c in cells):
            return False
        if ln.i in self.header_lines or ln.i in self.table_lines:
            return False
        if name_seen:
            for c in cells:
                if not (_SIGNED_RE.match(c) or name_score(c) >= 0.5 or _DATE_LINE_RE.match(c)
                        or not _has_letter(c) or (_SIG_TITLE_RE.search(c) and len(c) <= 60)):
                    return False
            return True
        if len(cells) >= 2:
            return all(len(c) <= 70 for c in cells) and not any(_APPENDIX_RE.match(c) for c in cells)
        s = ln.s
        if _APPENDIX_RE.match(s) or self._is_note(s) or self._title_followed_by_note(ln.i):
            return False
        lab = self.label(ln)
        if lab and lab.kind in HEADING_KINDS + ("clause",):
            return False
        if _SIGNED_RE.match(s) or name_score(s) >= 0.5 or _DATE_LINE_RE.match(s):
            return True
        if _SIG_TITLE_RE.search(s) and len(s) <= 70:
            return True
        if ln.indent >= 25 and len(s) <= 70:
            return True
        if self.upper(ln) and len(s) <= 60:
            return True
        if not _has_letter(s) and len(s) <= 10:
            return True
        return False

    def _sig_start_ish(self, ln: _Line) -> bool:
        cells = self._cell_texts(ln)
        if len(cells) >= 2:
            return all(len(c) <= 70 for c in cells)
        s = ln.s
        if len(s) > 80:
            return False
        if _SIG_TITLE_RE.search(s) or _SIGNED_RE.match(s):
            return True
        return ln.indent >= 25 or name_score(s) >= 0.8

    def scan_signature(self, k0: int):
        """Return (start_line, end_line) of a signature block starting at line k0, or None."""
        name_seen = False
        evidence = 0
        last = None
        k = k0
        while k < self.n:
            ln = self.lines[k]
            if ln.blank:
                k += 1
                continue
            if not self._sig_compatible(ln, name_seen):
                break
            cells = self._cell_texts(ln)
            for c in cells:
                if name_score(c) >= 0.5:
                    name_seen = True
                    evidence += 1
                elif _SIG_TITLE_RE.search(c) or _SIGNED_RE.match(c):
                    evidence += 1
            last = k
            k += 1
        if last is None or evidence == 0:
            return None
        if k >= self.n or name_seen:
            return (k0, last + 1)
        nxt = self.lines[k]
        if _RECIP_RE.match(nxt.s) or _BULLET_RE.match(nxt.s) or _APPENDIX_RE.match(nxt.s) \
                or self._title_followed_by_note(k) or nxt.i in self.header_lines:
            return (k0, last + 1)
        if any(_BULLET_RE.match(c) for c in self._cell_texts(nxt)):
            return (k0, last + 1)
        return None

    def _sig_line_spans(self, ln: _Line, region: str) -> list:
        """Spans for one line of a signature block (region 'sig') or post-signature line ('post')."""
        out = []
        pieces = ln.cells if len(ln.cells) >= 2 else [(ln.cstart, ln.cend)]
        for a, b in pieces:
            t = self.text(a, b)
            ns = name_score(t)
            flags = ["columnar"] if len(pieces) >= 2 else []
            if not _has_letter(t):
                out.append(_Span(a, b, "separator", False, 1.0, flags=flags))
            elif ns >= 0.8:
                out.append(_Span(a, b, "signature_name", False, ns, {"name": True}, flags=flags))
            elif region == "sig" or _SIGNED_RE.match(t) or _SIG_TITLE_RE.search(t) or ns >= 0.5:
                conf = 0.9 if region == "sig" else 0.6
                out.append(_Span(a, b, "signature", True, conf, flags=flags))
            elif _BULLET_RE.match(t):
                out.append(_Span(a, b, "list_item", True, 0.8, {"label": _BULLET_RE.match(t).group(1)},
                                 flags=flags))
            else:
                out.append(_Span(a, b, "unknown", True, 0.4, flags=flags))
        return out

    def _columnar_spans(self, bl: _Block, near_header: bool) -> list:
        ln = self.lines[bl.a]
        out = []
        for a, b in ln.cells:
            t = self.text(a, b)
            if not _has_letter(t):
                kind = "header_cell" if (near_header and _has_digit(t)) else "separator"
                out.append(_Span(a, b, kind, False, 1.0, flags=["columnar"]))
            elif near_header:
                out.append(_Span(a, b, "header_cell", True, 0.9, flags=["columnar"]))
            else:
                out.append(_Span(a, b, "unknown", True, 0.4, flags=["columnar"]))
        return out

    def _header_spans(self, bl: _Block) -> list:
        ln = self.lines[bl.a]
        pieces = ln.cells if len(ln.cells) >= 2 else [(ln.cstart, ln.cend)]
        out = []
        for a, b in pieces:
            t = self.text(a, b)
            if not _has_letter(t):
                kind = "header_cell" if _has_digit(t) else "separator"
                out.append(_Span(a, b, kind, False, 1.0))
            else:
                out.append(_Span(a, b, "header_cell", True, 1.0))
        return out

    def _table_spans(self, bl: _Block, start_group: int) -> tuple:
        a, b, kind, tid = next(t for t in self.tables if t[3] == bl.table_id)
        rows = []
        for k in range(a, b):
            ln = self.lines[k]
            if ln.blank:
                continue
            if kind == "pipe":
                rows.append(self._pipe_row(ln))
            else:
                cells = self._cells(ln, 2)
                row = []
                for col, (x, y) in enumerate(cells):
                    t = self.text(x, y)
                    if _has_letter(t):
                        sp = _Span(x, y, "table_cell", True, 0.9)
                        sp.table = {"numeric": False}
                    else:
                        sp = _Span(x, y, "table_cell", False, 1.0)
                        sp.table = {"numeric": _has_digit(t)}
                    sp.table["col"] = col
                    row.append(sp)
                rows.append(row)
        certain = False
        if kind == "pipe":
            counts = {len([s for s in r if s.kind == "table_cell" or s.flags == ["pipe_empty"]])
                      for r in rows if not any("table_separator_row" in s.flags for s in r)}
            widths = {r[0].table.get("width") for r in rows if r and r[0].table and r[0].table.get("width")}
            certain = len(widths) == 1 and len(counts) >= 1
        # groups <= TABLE_CAP tokens, row order kept
        gi = start_group
        cur_tok = 0
        out = []
        for r_i, row in enumerate(rows):
            rt = sum(self._tokens(make_payload(self.text(sp.s, sp.e))) for sp in row if sp.translatable)
            if out and cur_tok + rt > TABLE_CAP and cur_tok > 0:
                gi += 1
                cur_tok = 0
            cur_tok += rt
            for sp in row:
                tb = dict(sp.table or {})
                tb.pop("width", None)
                tb.update({"group": f"t{gi}", "table_id": tid, "row": r_i, "certain": certain})
                tb.setdefault("numeric", False)
                tb.setdefault("col", -1)
                sp.table = tb
                if not certain and "broken_table" not in sp.flags:
                    sp.flags.append("broken_table")
                out.append(sp)
        return out, gi + 1

    def _pipe_row(self, ln: _Line) -> list:
        s = ln.s
        if _PIPE_SEP_RE.match(s):
            sp = _Span(ln.cstart, ln.cend, "separator", False, 1.0, flags=["table_separator_row"])
            sp.table = {"numeric": False, "col": -1}
            return [sp]
        out = []
        txt = ln.text
        # split on unescaped pipes
        pos = [m.start() for m in re.finditer(r"(?<!\\)\|", txt)]
        width = len(pos) - 1
        for k, p in enumerate(pos):
            sp = _Span(ln.start + p, ln.start + p + 1, "separator", False, 1.0, flags=["table_pipe"])
            sp.table = {"numeric": False, "col": -1}
            out.append(sp)
            if k + 1 < len(pos):
                seg = txt[p + 1:pos[k + 1]]
            else:
                seg = txt[p + 1:]
            if seg.strip():
                lo = p + 1 + (len(seg) - len(seg.lstrip()))
                hi = p + 1 + len(seg.rstrip())
                t = txt[lo:hi]
                if _has_letter(t):
                    cs = _Span(ln.start + lo, ln.start + hi, "table_cell", True, 1.0)
                    cs.table = {"numeric": False, "col": k}
                else:
                    cs = _Span(ln.start + lo, ln.start + hi, "table_cell", False, 1.0)
                    cs.table = {"numeric": _has_digit(t), "col": k}
                out.append(cs)
        if out:
            out[0].table["width"] = width
        return out

    def _tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.tok is not None:
            return self.tok.count(text)
        return math.ceil(len(text) / 3)

    # -- main --------------------------------------------------------------------------------
    def run(self) -> list:
        self.find_headers()
        self.find_tables()
        blocks = self.segment()
        for bl in blocks:
            self.classify(bl)
        blocks = self.join_page_breaks(blocks)
        spans = self.structure(blocks)
        return spans

    def structure(self, blocks: list) -> list:
        """Signature blocks, scopes, quotes, paths, protected -> ordered spans."""
        # pass A: signature regions and scope starts (line based on top of blocks)
        sig_lines: dict = {}           # line index -> "sig"
        scope_of_block: list = []
        scope = "main"
        n_att = n_app = 0
        body_seen = False
        sig_seen_any = False
        after_sig = False              # directly after a signature block (only blanks/recipients since)
        first_title_pending = False    # next title block of a new scope is its attachment_title
        new_scope_blocks: set = set()
        bi = 0
        processed: list[_Block] = []
        while bi < len(blocks):
            bl = blocks[bi]
            first = self.lines[bl.a]
            # scope boundary checks
            if bl.typ in ("text", "columnar", "header") and sig_seen_any or bl.typ == "text" and body_seen:
                start_new = None
                if bl.typ == "text" and sig_seen_any and _APPENDIX_RE.match(first.s) and len(first.s) <= 120:
                    start_new = "appendix"
                elif bl.typ == "text" and after_sig and bl.kind == "title":
                    start_new = "attachment"
                elif bl.typ == "text" and body_seen and bl.kind == "title" and self._title_followed_by_note(bl.a):
                    start_new = "attachment"
                if start_new == "appendix":
                    n_app += 1
                    scope = f"appendix_{n_app}"
                elif start_new == "attachment":
                    n_att += 1
                    scope = f"attachment_{n_att}"
                if start_new:
                    body_seen = False
                    after_sig = False
                    first_title_pending = True
                    new_scope_blocks.add(id(bl))
            # signature detection
            if bl.typ in ("text", "columnar") and body_seen and bl.kind not in HEADING_KINDS \
                    and self._sig_start_ish(first):
                rng = self.scan_signature(bl.a)
                if rng is not None:
                    s0, s1 = rng
                    # absorb every block that overlaps [s0, s1)
                    j = bi
                    while j < len(blocks) and blocks[j].a < s1:
                        j += 1
                    sig_block = _Block(s0, s1, typ="sig", kind="signature", scope_id=scope)
                    # a block that started before s0 cannot exist (s0 == bl.a); trailing part of the last
                    # absorbed block after s1 (if any) is re-queued as its own block
                    tail = None
                    last_abs = blocks[j - 1]
                    if last_abs.b > s1:
                        tail = _Block(s1, last_abs.b, typ=last_abs.typ, kind=last_abs.kind,
                                      label=None, flags=list(last_abs.flags))
                    processed.append(sig_block)
                    scope_of_block.append(scope)
                    for k in range(s0, s1):
                        sig_lines[k] = "sig"
                    rest = blocks[j:]
                    if tail is not None:
                        # re-segment the tail lines as fresh text blocks
                        tail_blocks = self._resegment(tail.a, tail.b)
                        rest = tail_blocks + rest
                    blocks = blocks[:bi] + [sig_block] + rest
                    bi += 1
                    sig_seen_any = True
                    after_sig = True
                    body_seen = False
                    continue
            if bl.typ == "text" and bl.kind in BODY_KINDS:
                body_seen = True
            if bl.typ == "text" and bl.kind in ("paragraph", "clause", "point", "article", "title", "note") \
                    and not _RECIP_RE.match(first.s):
                if bl.kind not in ("title", "note"):
                    after_sig = False
            if bl.typ == "table":
                body_seen = True
                after_sig = False
            bl.scope_id = scope
            processed.append(bl)
            scope_of_block.append(scope)
            bi += 1
        blocks = processed
        for bl in blocks:
            if id(bl) in new_scope_blocks and bl.typ == "text" and bl.kind in ("title", "paragraph", "note") \
                    and bl.kind != "note":
                bl.kind = "attachment_title"

        # pass B: quote scopes on text blocks (memory only)
        self._quotes(blocks)

        # pass C: spans with paths/protected
        spans: list[_Span] = []
        stacks: dict = {}
        quote_saved: dict = {}
        group_counter = 1
        hdr_near = self._header_neighbourhood()
        post_sig = False
        prev_scope = None
        for bl in blocks:
            scope = bl.scope_id
            base_scope = getattr(bl, "_outer_scope", scope)
            if base_scope != prev_scope:
                post_sig = False
            prev_scope = base_scope
            st = stacks.setdefault(base_scope, {})
            if scope.startswith("quote_"):
                qst = quote_saved.setdefault(scope, dict(st))
                cur_stack = qst
            else:
                cur_stack = st
            path = self._path_of(cur_stack)
            if bl.typ == "header":
                for sp in self._header_spans(bl):
                    sp.scope_id, sp.path = scope, path
                    spans.append(sp)
                continue
            if bl.typ == "table":
                tsp, group_counter = self._table_spans(bl, group_counter)
                for sp in tsp:
                    sp.scope_id, sp.path = scope, path
                    spans.append(sp)
                continue
            if bl.typ == "sig":
                for k in range(bl.a, bl.b):
                    ln = self.lines[k]
                    if ln.blank:
                        continue
                    for sp in self._sig_line_spans(ln, "sig"):
                        sp.scope_id, sp.path = scope, path
                        spans.append(sp)
                post_sig = True
                continue
            if bl.typ == "columnar":
                ln = self.lines[bl.a]
                if post_sig and bl.a not in self.header_lines and not hdr_near.get(bl.a):
                    sps = self._sig_line_spans(ln, "post")
                else:
                    sps = self._columnar_spans(bl, bool(hdr_near.get(bl.a)))
                for sp in sps:
                    sp.scope_id, sp.path = scope, path
                    spans.append(sp)
                continue
            # text block
            s0 = self.lines[bl.a].cstart
            s1 = self.lines[bl.b - 1].cend
            kind = bl.kind
            prot: dict = {}
            lab = bl.label
            if lab is not None:
                self._push(cur_stack, lab)
                path = self._path_of(cur_stack, include_point=True)
                if lab.kind in HEADING_KINDS:
                    prot = {"heading_type": lab.kind, "heading_number": lab.number}
                elif lab.kind in ("clause", "point"):
                    prot = {"label": lab.label}
                elif lab.kind == "list_item":
                    prot = {"label": lab.label}
                if lab.prefix and prot:
                    prot["prefix"] = lab.prefix.strip()
            elif kind in ("paragraph", "unknown"):
                path = self._path_of(cur_stack, include_point=False)
            translatable = kind != "separator"
            conf = bl.conf
            if kind == "unknown":
                conf = min(conf, 0.4)
            sp = _Span(s0, s1, kind, translatable, conf, prot, None, list(bl.flags), scope, path)
            spans.append(sp)
        return spans

    def _resegment(self, a: int, b: int) -> list:
        out = []
        cur = None
        for k in range(a, b):
            ln = self.lines[k]
            if ln.blank:
                cur = None
                continue
            if len(ln.cells) >= 2:
                cur = None
                out.append(_Block(k, k + 1, typ="columnar", kind="unknown"))
                continue
            if cur is not None and not self.split_between(self.lines[cur.b - 1], ln):
                cur.b = k + 1
                continue
            cur = _Block(k, k + 1)
            out.append(cur)
        for bl in out:
            self.classify(bl)
        return out

    def _header_neighbourhood(self) -> dict:
        """Line index -> True when a columnar line sits within 3 lines of a header line or motto."""
        near = {}
        anchors = [ln.i for ln in self.lines if not ln.blank and (ln.i in self.header_lines
                                                                   or _MOTTO_RE.search(ln.s))]
        for i in anchors:
            for k in range(max(0, i - 3), min(self.n, i + 4)):
                near[k] = True
        return near

    # -- paths -------------------------------------------------------------------------------
    _LEVELS = ("part", "chapter", "section", "article", "item", "clause", "subclause", "point")

    def _push(self, stack: dict, lab: _Label):
        lv = lab.level
        if lv == "bullet":
            return
        idx = self._LEVELS.index(lv)
        for deeper in self._LEVELS[idx:]:
            stack.pop(deeper, None)
        stack[lv] = lab.path_label

    def _path_of(self, stack: dict, include_point: bool = False) -> list:
        out = []
        for lv in self._LEVELS:
            if lv == "point" and not include_point:
                continue
            if lv in stack:
                out.append(stack[lv])
        return out

    # -- quotes ------------------------------------------------------------------------------
    @staticmethod
    def _quote_scan(text: str, depth: int):
        """Advance quote depth over text. Returns (depth_after, closed_at_end, zero_inside)."""
        zero_inside = False
        last_close = -1
        n = len(text)
        for i, ch in enumerate(text):
            if ch == "“":
                depth += 1
            elif ch == "”":
                if depth > 0:
                    depth -= 1
                    last_close = i
                    if depth == 0 and text[i + 1:].strip(" \t\n.,;:)]”\"") != "":
                        zero_inside = True
            elif ch == '"':
                prev = text[i - 1] if i > 0 else " "
                nxt = text[i + 1] if i + 1 < n else " "
                opener = (prev.isspace() or prev in "([{") and not nxt.isspace()
                if opener and not (depth > 0 and prev.isalnum()):
                    depth += 1
                elif depth > 0:
                    depth -= 1
                    last_close = i
                    if depth == 0 and text[i + 1:].strip(" \t\n.,;:)]”\"") != "":
                        zero_inside = True
        closed_at_end = depth == 0 and last_close >= 0 and text[last_close + 1:].strip(" \t\n.,;:)]”\"") == ""
        return depth, closed_at_end, zero_inside

    def _quotes(self, blocks: list):
        n_q = 0
        i = 0
        while i < len(blocks):
            bl = blocks[i]
            if bl.typ != "text" or bl.scope_id.startswith("quote_"):
                i += 1
                continue
            txt = self.block_text(bl)
            if not txt[:1] in ("“", '"'):
                i += 1
                continue
            depth, closed, zero_inside = self._quote_scan(txt, 0)
            if zero_inside and not (closed and depth == 0):
                i += 1
                continue
            j = i
            ok = closed and depth == 0
            if not ok:
                k = i + 1
                while k < len(blocks) and k - i <= 400:
                    nb = blocks[k]
                    if nb.typ in ("sig", "header") or nb.scope_id != bl.scope_id:
                        break
                    if nb.typ == "text" and nb.kind in ("chapter", "part"):
                        break
                    if nb.typ == "text":
                        depth, closed, _ = self._quote_scan(self.block_text(nb), depth)
                        if depth == 0:
                            ok = closed
                            j = k
                            break
                    k += 1
            if not ok:
                i += 1
                continue
            n_q += 1
            qid = f"quote_{n_q}"
            for k in range(i, j + 1):
                blocks[k]._outer_scope = blocks[k].scope_id
                blocks[k].scope_id = qid
            i = j + 1

# ----------------------------------------------------------------------------------------------
# spans -> nodes
# ----------------------------------------------------------------------------------------------
def _spans_to_nodes(src: str, spans: list) -> list:
    spans = sorted(spans, key=lambda sp: (sp.s, sp.e))
    nodes: list[Node] = []
    prev_end = 0
    pending_lead_start = 0
    for idx, sp in enumerate(spans):
        if sp.s < prev_end:
            raise ValueError(f"overlapping spans at {sp.s} < {prev_end}")
        gap = src[prev_end:sp.s]
        if gap.strip():
            raise ValueError(f"uncovered text at {prev_end}: {gap[:40]!r}")
        if nodes:
            cut = gap.rfind("\n")
            if cut >= 0:
                trail_add, lead = gap[:cut + 1], gap[cut + 1:]
            else:
                trail_add, lead = "", gap
            last = nodes[-1]
            last.trail += trail_add
            last.raw_end += len(trail_add)
            lead_start = prev_end + len(trail_add)
        else:
            lead = gap
            lead_start = 0
        content = src[sp.s:sp.e]
        n = Node(node_id=f"n{len(nodes):05d}", kind=sp.kind, raw_start=lead_start, raw_end=sp.e, lead=lead,
                 content_raw=content, trail="", payload="", translatable=sp.translatable,
                 scope_id=sp.scope_id, path=list(sp.path), confidence=round(float(sp.confidence), 3),
                 protected=dict(sp.protected), table=sp.table, flags=list(sp.flags))
        if n.translatable:
            n.payload = make_payload(content)
            if not n.payload.strip():
                n.translatable = False
                n.payload = ""
        nodes.append(n)
        prev_end = sp.e
    tail = src[prev_end:]
    if tail.strip():
        raise ValueError("uncovered text at end")
    if nodes:
        nodes[-1].trail += tail
        nodes[-1].raw_end += len(tail)
    return nodes

def _fallback_nodes(src: str, reason: str) -> list:
    """Last-resort parse (bug safety net): blank-line separated paragraphs, low confidence, flagged."""
    spans = []
    for m in re.finditer(r"\S(?:.*\S)?(?:\n[ \t]*\S.*)*", src):
        seg = m.group(0)
        s = m.start()
        e = s + len(seg.rstrip())
        kind = "paragraph" if _has_letter(seg) else "separator"
        spans.append(_Span(s, e, kind, kind != "separator", 0.3, flags=["fallback_parse"]))
    nodes = _spans_to_nodes(src, spans)
    if nodes:
        nodes[0].flags.append("fallback_reason:" + reason[:120])
    return nodes

def check_partition(src: str, nodes: list) -> list:
    """Invariant problems (empty list = ok): exact roundtrip, offsets, contiguity, coverage."""
    errs = []
    if "".join(n.raw for n in nodes) != src:
        errs.append("roundtrip")
    pos = 0
    for n in nodes:
        if n.raw_start != pos:
            errs.append(f"gap_or_overlap:{n.node_id}")
            break
        if src[n.raw_start:n.raw_end] != n.raw:
            errs.append(f"offset:{n.node_id}")
            break
        if n.lead.strip() or n.trail.strip():
            errs.append(f"non_ws_outside_content:{n.node_id}")
            break
        if n.translatable and not n.payload:
            errs.append(f"empty_payload:{n.node_id}")
            break
        if not n.translatable and n.payload:
            errs.append(f"copy_with_payload:{n.node_id}")
            break
        if n.kind not in NODE_KINDS:
            errs.append(f"bad_kind:{n.node_id}")
            break
        pos = n.raw_end
    if nodes and pos != len(src):
        errs.append("end_offset")
    if not nodes and src:
        errs.append("no_nodes")
    return errs

# ----------------------------------------------------------------------------------------------
# public API
# ----------------------------------------------------------------------------------------------
def parse_document(src: str, doc_id: str, src_lang: str, tok=None) -> list:
    """Parse one SOURCE document into nodes that partition `src` exactly."""
    if src is None or src == "":
        return []
    if not src.strip():
        return [Node(node_id="n00000", kind="separator", raw_start=0, raw_end=len(src), lead=src,
                     content_raw="", trail="", payload="", translatable=False)]
    try:
        spans = _Parser(src, doc_id, src_lang, tok).run()
        nodes = _spans_to_nodes(src, spans)
        errs = check_partition(src, nodes)
        if errs:
            raise ValueError("invariant:" + ",".join(errs[:3]))
    except Exception as exc:  # never lose text: fall back and flag it
        nodes = _fallback_nodes(src, repr(exc))
    return nodes

def reconstruct(nodes) -> str:
    return "".join(n.raw for n in nodes)

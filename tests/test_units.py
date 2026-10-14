"""Tests for the unit steps (s03 split/pack, s05 budget, s06 parse_output, s08 recovery) (packing, splitting, markers, payload, parse_output, budget, recovery).

Nodes are built by hand (factory below) or from SOURCE docs with a crude test-only paragraph
splitter; parse.py is only used by the optional integration test at the end."""
import hashlib
import random
import re
import statistics

import pytest

from types import SimpleNamespace

from docmt.steps import s03_pack, s03_split, s05_budget, s06_parse_output, s08_recovery

# The unit steps live in these step modules; the tests address them as `U.<name>`.
U = SimpleNamespace(**{n: getattr(m, n) for m in (s03_split, s03_pack, s05_budget, s06_parse_output, s08_recovery)
                       for n in dir(m) if not n.startswith("__")})
from docmt.core.config import RunConfig
from docmt.core.datatypes import Node, Unit

CFG = RunConfig()
SIG_KINDS = ("signature", "signature_name")


# ----------------------------------------------------------------------------------- helpers
def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def T(content, **kw):
    return dict(content=content, translatable=True, **kw)


def C(content, **kw):
    return dict(content=content, translatable=False, **kw)


def build(specs):
    """Node list from specs (dicts: content, translatable, kind, scope, table, lead, trail, protected).
    Respects the Node invariants: offsets contiguous, raw == lead + content_raw + trail."""
    nodes, parts, pos = [], [], 0
    for i, s in enumerate(specs):
        lead = s.get("lead", "  ")
        trail = s.get("trail", "\n\n")
        content = s["content"]
        raw = lead + content + trail
        tr = s["translatable"]
        nodes.append(Node(
            node_id=f"n{i:05d}", kind=s.get("kind", "paragraph" if tr else "separator"),
            raw_start=pos, raw_end=pos + len(raw), lead=lead, content_raw=content, trail=trail,
            payload=U.payload_of(content) if tr else "", translatable=tr,
            scope_id=s.get("scope", "main"), protected=dict(s.get("protected", {})),
            table=s.get("table"), flags=[]))
        parts.append(raw)
        pos += len(raw)
    src = "".join(parts)
    assert "".join(n.raw for n in nodes) == src
    return src, nodes


WORDS = ("the agency shall submit report within working period after receipt according regulations "
         "ministry justice organization individual decision provincial committee responsible "
         "implementation examiner allowance assessment forensic document request").split()


def sentence(rng, n):
    s = " ".join(rng.choice(WORDS) for _ in range(n))
    return s[0].upper() + s[1:] + "."


def para(rng, n_sent, lo=6, hi=18):
    return " ".join(sentence(rng, rng.randint(lo, hi)) for _ in range(n_sent))


def wrap(text, width=90, indent="  "):
    """Hard-wrap like the PDF-to-text sources (continuation lines indented)."""
    lines, cur = [], ""
    for w in text.split(" "):
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    lines.append(cur)
    return ("\n" + indent).join(lines)


def run_key(n):
    if n.table is not None or n.kind == "table_cell":
        return ("table", n.scope_id, (n.table or {}).get("group"))
    if n.kind in SIG_KINDS:
        return ("signature", n.scope_id, None)
    return ("text", n.scope_id, None)


LIMITS = {"text": (CFG.source_cap, CFG.max_nodes), "table": (CFG.table_cap, CFG.max_table_nodes),
          "signature": (CFG.signature_cap, CFG.max_nodes)}


def check_units(units, nodes, src, tok, ns, allow_single=True):
    """All packing invariants of the unit steps."""
    pos = {n.node_id: i for i, n in enumerate(nodes)}
    trans = [n.node_id for n in nodes if n.translatable]
    sent = [nid for u in units for nid in u.node_ids]
    assert sent == trans                                   # each translatable node once, in order
    ids = [u.unit_id for u in units]
    assert len(set(ids)) == len(ids)
    for k, u in enumerate(units):
        assert u.unit_id == f"{u.doc_id}:u{k:04d}" and u.split_from is None
        un = [nodes[pos[nid]] for nid in u.node_ids]
        assert all(n.translatable for n in un)              # copy nodes never sent
        assert u.payload == U.render_payload(un, u.markers)
        assert u.source_tokens == tok.count(u.payload)
        if len(un) == 1:
            assert u.markers == []
        else:
            assert u.markers == [U.marker(ns, pos[n.node_id]) for n in un]
        assert (u.raw_start, u.raw_end) == (un[0].raw_start, un[-1].raw_end)
        assert u.raw_sha256 == sha(src[u.raw_start:u.raw_end])
        keys = {run_key(n) for n in un}
        assert len(keys) == 1, (u.unit_id, keys)            # one scope / table group / block kind
        key = keys.pop()
        assert u.scope_id == key[1]
        first, last = pos[un[0].node_id], pos[un[-1].node_id]
        for n in nodes[first:last + 1]:                     # nothing structural inside the span
            if n.translatable:
                assert n.node_id in u.node_ids
            else:
                kk = run_key(n)
                assert kk == key or (kk[0] == "text" and kk[1] == key[1]), (u.unit_id, n.node_id)
        if u.kind == "single":
            assert allow_single and len(un) == 1
        else:
            assert u.kind == key[0]
            cap, max_n = LIMITS[u.kind]
            assert len(un) <= max_n and u.source_tokens <= cap, (u.unit_id, len(un), u.source_tokens)


def paragraph_nodes(src):
    """Test-only crude node list: blank-line separated blocks (letters -> translatable)."""
    blocks, off, cur = [], 0, None
    for ln in src.splitlines(keepends=True):
        if ln.strip():
            cur = [off, off + len(ln)] if cur is None else [cur[0], off + len(ln)]
        elif cur:
            blocks.append(tuple(cur))
            cur = None
        off += len(ln)
    if cur:
        blocks.append(tuple(cur))
    nodes = []
    for k, (s, _e) in enumerate(blocks):
        start = 0 if k == 0 else s
        end = blocks[k + 1][0] if k + 1 < len(blocks) else len(src)
        text = src[start:end]
        body = text.lstrip()
        lead = text[:len(text) - len(body)]
        content = body.rstrip()
        trail = body[len(content):]
        tr = bool(re.search(r"[^\W\d_]", content))
        nodes.append(Node(node_id=f"n{k:05d}", kind="paragraph" if tr else "separator",
                          raw_start=start, raw_end=end, lead=lead, content_raw=content, trail=trail,
                          payload=U.payload_of(content) if tr else "", translatable=tr))
    assert "".join(n.raw for n in nodes) == src
    return nodes


@pytest.fixture(scope="session")
def smoke_nodes(smoke6):
    return [(row, src, paragraph_nodes(src)) for row, src in smoke6]


# ----------------------------------------------------------------------------------- markers
def test_marker_format():
    assert U.marker("SEG", 5) == "[SEG_000005]"
    assert U.marker("SEG0A1F", 123456) == "[SEG0A1F_123456]"
    assert U.MARKER_RE.fullmatch(U.marker("SEG", 0))
    for bad in (-1, 1_000_000):
        with pytest.raises(ValueError):
            U.marker("SEG", bad)
    for bad_ns in ("seg", "SEGXYZ1", "SEG12345", "TAG"):
        with pytest.raises(ValueError):
            U.marker(bad_ns, 1)


def test_namespace_default_and_collision():
    plain = "Article 1. Scope\nThis Decree applies to [Note 1] and SEG_ codes."
    assert U.marker_namespace(plain) == "SEG"
    src = "Text that already contains a marker [SEG_000001] from the source."
    ns = U.marker_namespace(src)
    assert ns == "SEG" + hashlib.sha256(src.encode("utf-8")).hexdigest()[:4].upper()
    assert re.fullmatch(r"SEG[0-9A-F]{4}", ns)
    assert f"[{ns}_" not in src and U.marker_namespace(src) == ns          # deterministic
    assert U.MARKER_RE.fullmatch(U.marker(ns, 3)).group(1) == ns
    # a Cyrillic homoglyph would become "[SEG_" after normalize(): treated as a collision too
    homo = "Văn bản [SЕG_000002] có ký tự lạ."
    assert "[SEG_" not in homo and U.marker_namespace(homo) != "SEG"


def test_render_payload(tok):
    _, nodes = build([T("First clause."), T("Second clause."), T("Third.")])
    assert U.render_payload(nodes[:1], []) == nodes[0].payload
    marks = [U.marker("SEG", i) for i in range(3)]
    assert U.render_payload(nodes, marks) == (
        "[SEG_000000] First clause.\n[SEG_000001] Second clause.\n[SEG_000002] Third.")
    with pytest.raises(ValueError):
        U.render_payload(nodes, marks[:2])
    with pytest.raises(ValueError):
        U.render_payload([], [])


# ----------------------------------------------------------------------------------- packing
def test_pack_text_limits_synthetic(tok):
    rng = random.Random(7)
    specs = []
    for i in range(70):
        n_sent = rng.choice([1, 1, 1, 2, 3, 6, 12, 20])
        specs.append(T(wrap(para(rng, n_sent))))
        if i % 9 == 4:
            specs.append(C("-" * rng.randint(5, 30)))      # neutral copy line inside the run
    src, nodes = build(specs)
    nodes = U.split_long_nodes(nodes, tok, CFG)
    units = U.pack_units(nodes, "Cat/1", tok, CFG, "SEG")
    check_units(units, nodes, src, tok, "SEG", allow_single=False)
    assert any(len(u.node_ids) == CFG.max_nodes for u in units)           # node limit reached
    assert any(u.source_tokens > CFG.source_cap * 0.8 for u in units)      # token cap reached
    for a, b in zip(units, units[1:]):                                     # greedy: next node did not fit
        pos = {n.node_id: i for i, n in enumerate(nodes)}
        nxt = nodes[pos[b.node_ids[0]]]
        grown = [nodes[pos[x]] for x in a.node_ids] + [nxt]
        marks = [U.marker("SEG", pos[n.node_id]) for n in grown]
        assert (len(grown) > CFG.max_nodes
                or tok.count(U.render_payload(grown, marks)) > CFG.source_cap)


def boundary_doc():
    rng = random.Random(11)
    t1 = lambda r, c: {"group": "t1", "row": r, "col": c, "numeric": False, "certain": False}
    t2 = lambda r, c: {"group": "t2", "row": r, "col": c, "numeric": False, "certain": True}
    specs = [
        T("THE MINISTRY OF HEALTH", kind="header_cell", trail=" " * 9),
        T("SOCIALIST REPUBLIC OF VIETNAM", kind="header_cell", lead="", trail="\n"),
        C("-------", lead="              ", trail="\n"),
        T("No. 31/2015/TT-BYT", kind="header_cell", trail=" " * 20),
        T("Hanoi, October 14, 2015", kind="header_cell", lead="", trail="\n\n"),
        T("CIRCULAR", kind="title"),
        T(wrap(para(rng, 2))),
        T(wrap(para(rng, 1))),
        T("Table 01", kind="paragraph"),
    ]
    for r in range(3):                                   # table t1: text cells + numeric copy cells
        specs += [T(f"Type {r} examination of the body", table=t1(r, 0), kind="table_cell", trail="   "),
                  C(f"0{r + 1}", table={**t1(r, 1), "numeric": True}, kind="table_cell", trail="   "),
                  T("Allowance per working day", table=t1(r, 2), kind="table_cell", lead="", trail="\n")]
    specs += [C("|---|---|", kind="separator", trail="\n")]          # neutral line inside the table
    specs += [T(f"Extra row {r}", table=t1(3 + r, 0), kind="table_cell", trail="\n") for r in range(2)]
    specs += [T(wrap(para(rng, 1)))]                                 # text between tables
    specs += [C("16", table={**t2(0, 0), "numeric": True}, kind="table_cell", trail="\n")]  # numeric-only table
    specs += [T(wrap(para(rng, 1)))]
    specs += [T(f"Cell {r}", table=t1(9, r), kind="table_cell", trail="\n") for r in range(2)]
    specs += [T(f"Other {r}", table=t2(1, r), kind="table_cell", trail="\n") for r in range(2)]  # adjacent group
    specs += [T("ON BEHALF OF THE MINISTER", kind="signature"),
              T("DEPUTY MINISTER", kind="signature"),
              C("Nguyen Viet Tien", kind="signature_name", protected={"name": True})]
    specs += [T("REGULATIONS", kind="attachment_title", scope="attachment_1"),
              T("(Promulgated together with Circular No. 31/2015/TT-BYT)", kind="note", scope="attachment_1"),
              T(wrap(para(rng, 3)), scope="attachment_1"),
              T("MINISTER", kind="signature", scope="attachment_1"),
              T(wrap(para(rng, 1)), scope="attachment_1"),
              T("(Signed)", kind="signature", scope="attachment_1")]
    return build(specs)


def test_pack_boundaries(tok):
    src, nodes = boundary_doc()
    units = U.pack_units(nodes, "Cat/2", tok, CFG, "SEG")
    check_units(units, nodes, src, tok, "SEG", allow_single=False)
    where = {nid: u.unit_id for u in units for nid in u.node_ids}
    by_content = {n.content_raw: n.node_id for n in nodes}
    # header cells and title packed together across the neutral separator
    assert where[by_content["THE MINISTRY OF HEALTH"]] == where[by_content["CIRCULAR"]]
    # text before / after a table never share a unit, even a table of copy-only cells
    texts = [n.node_id for n in nodes if n.kind == "paragraph" and n.translatable]
    assert len({where[x] for x in texts}) >= 3
    # table cells of t1 split by the neutral line stay in one table unit; t1 / t2 never mixed
    t1_units = {where[n.node_id] for n in nodes if n.table and n.table["group"] == "t1" and n.translatable
                and n.table["row"] < 9}
    assert len(t1_units) == 1
    kinds = {u.unit_id: u.kind for u in units}
    assert {kinds[where[n.node_id]] for n in nodes if n.table and n.translatable} == {"table"}
    sig = [u for u in units if u.kind == "signature"]
    assert [len(u.node_ids) for u in sig] == [2, 1, 1]           # main block, two attachment blocks
    assert all(n.node_id not in where for n in nodes if not n.translatable)
    assert {u.scope_id for u in units} == {"main", "attachment_1"}


def test_pack_table_and_signature_caps(tok):
    rng = random.Random(3)
    tg = lambda r, c: {"group": "t9", "row": r, "col": c, "numeric": False, "certain": True}
    specs = [T(" ".join(rng.choice(WORDS) for _ in range(rng.randint(2, 30))), table=tg(i // 3, i % 3),
               kind="table_cell", trail="\n") for i in range(60)]
    specs += [T(" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 25))).upper(), kind="signature",
                trail="\n") for _ in range(14)]
    src, nodes = build(specs)
    units = U.pack_units(nodes, "Cat/3", tok, CFG, "SEG")
    check_units(units, nodes, src, tok, "SEG", allow_single=False)
    tab = [u for u in units if u.kind == "table"]
    sig = [u for u in units if u.kind == "signature"]
    assert len(tab) >= 2 and max(len(u.node_ids) for u in tab) == CFG.max_table_nodes
    assert all(u.source_tokens <= CFG.table_cap for u in tab)
    assert len(sig) >= 2 and all(u.source_tokens <= CFG.signature_cap for u in sig)


def test_pack_oversize_node_is_single(tok):
    rng = random.Random(5)
    big = wrap(para(rng, 100))
    src, nodes = build([T("Short."), T(big), T("Tail.")])
    assert tok.count(nodes[1].payload) > CFG.source_cap
    units = U.pack_units(nodes, "Cat/4", tok, CFG, "SEG")
    check_units(units, nodes, src, tok, "SEG", allow_single=True)
    assert [u.kind for u in units] == ["text", "single", "text"]
    assert units[1].markers == [] and units[1].payload == nodes[1].payload


def test_pack_rejects_broken_offsets(tok):
    _, nodes = build([T("One."), T("Two.")])
    nodes[1].raw_start += 1
    with pytest.raises(ValueError):
        U.pack_units(nodes, "Cat/5", tok, CFG, "SEG")


def test_pack_real_sources_after_split(tok, smoke_nodes):
    """Smoke6 SOURCE docs (crude paragraph nodes): every cap holds after per-kind splitting."""
    for row, src, nodes in smoke_nodes:
        ns = U.marker_namespace(src)
        split = U.split_long_nodes(nodes, tok, CFG)
        assert "".join(n.raw for n in split) == src
        units = U.pack_units(split, row["sample_id"], tok, CFG, ns)
        check_units(units, split, src, tok, ns, allow_single=False)


# ------------------------------------------------------------------------------------ splitting
PROTECTED = ["1.000.000", "1,000,000", "3.5", "2.1.", "No. 12/2011/TT-BTP", "VND 5,000,000", "15 days",
             "October 14, 2015", "Article 7", "5.000.000 đồng", "REGULA-\n  TION"]
ABBREV_TAILS = ("No.", "Nos.", "Art.", "e.g.", "i.e.", "etc.", "Mr.", "Dr.")


def protected_paragraph(rng, reps=6):
    chunks = []
    for _ in range(reps):
        chunks += [
            sentence(rng, 9),
            "The fine is VND 5,000,000 for each violation under Decision No. 12/2011/TT-BTP of the ministry.",
            sentence(rng, 7),
            "A sum of 1.000.000 or 1,000,000 shall be paid within 15 days, e.g. Hanoi applies Art. 5 rules.",
            "The rate is 3.5 percent as set in point 2.1. Other costs, etc. Mr. Nguyen signs on October 14, 2015.",
            "Clause 4 of Article 7 sets 5.000.000 đồng and the REGULA-TION table, i.e. the annex in force.",
            sentence(rng, 12),
        ]
    return wrap(" ".join(chunks)).replace("REGULA-TION ", "REGULA-\n  TION ")


def cut_positions(pieces):
    out, p = [], 0
    for x in pieces[:-1]:
        p += len(x.content_raw)
        out.append(p)
    return out


@pytest.mark.parametrize("cap", [20, 32, 48, 96])
def test_split_long_node_invariants_and_protected(tok, cap):
    rng = random.Random(cap)
    content = protected_paragraph(rng)
    assert "REGULA-\n  TION" in content
    src, nodes = build([T("Before."), T(content, kind="article",
                                          protected={"heading_type": "article", "heading_number": "3"}),
                        T("After.")])
    node = nodes[1]
    out = U.split_long_nodes(nodes, tok, cap)
    pieces = [p for p in out if U.root_node_id(p.node_id) == node.node_id]
    assert out[0] is nodes[0] and out[-1] is nodes[2]            # short nodes untouched
    assert len(pieces) >= 2
    assert [p.node_id for p in pieces] == [f"{node.node_id}.{i}" for i in range(1, len(pieces) + 1)]
    assert "".join(p.content_raw for p in pieces) == node.content_raw
    assert "".join(p.raw for p in pieces) == node.raw
    assert pieces[0].lead == node.lead and pieces[-1].trail == node.trail
    assert all(p.lead == "" for p in pieces[1:]) and all(p.trail == "" for p in pieces[:-1])
    assert pieces[0].raw_start == node.raw_start and pieces[-1].raw_end == node.raw_end
    for a, b in zip(pieces, pieces[1:]):
        assert a.raw_end == b.raw_start
    for p in pieces:
        assert src[p.raw_start:p.raw_end] == p.raw
        assert p.payload and tok.count(p.payload) <= cap
        assert p.kind == "article" and p.scope_id == node.scope_id and "split_piece" in p.flags
        assert "split_forced" not in p.flags
    assert pieces[0].protected == node.protected and all(p.protected == {} for p in pieces[1:])
    assert "".join(n.raw for n in out) == src
    cuts = cut_positions(pieces)
    for pat in PROTECTED:
        for m in re.finditer(re.escape(pat), content):
            assert not any(m.start() < c < m.end() for c in cuts), (pat, cap)
    for c in cuts:
        before = content[:c].rstrip()
        assert not before.endswith(ABBREV_TAILS), (before[-30:], cap)
        assert not content[c - 1].isalnum() or content[c - 1] == " "


def test_split_prefers_sentence_then_clause_then_space(tok):
    s1 = "The first sentence states the scope of this regulation clearly."
    s2 = "The second sentence, which is longer, names the responsible agency."
    s3 = "The third sentence ends the paragraph with the final provision."
    content = wrap(f"{s1} {s2} {s3}")
    cap = tok.count(U.payload_of(wrap(f"{s1} {s2}"))) + 1
    _, nodes = build([T(content)])
    pieces = U.split_node(nodes[0], tok, cap)
    assert pieces[0].content_raw.rstrip().endswith("agency.")
    # one long sentence with commas: clause boundary
    clause = ", ".join(f"the {w} of the provincial committee" for w in WORDS[:14]) + " applies."
    _, nodes = build([T(clause)])
    cap = tok.count(nodes[0].payload) // 2
    pieces = U.split_node(nodes[0], tok, cap)
    assert len(pieces) >= 2 and all(p.content_raw.rstrip().endswith(",") for p in pieces[:-1])
    # no punctuation at all: whitespace nearest to the cap
    flat = " ".join(WORDS * 3)
    _, nodes = build([T(flat)])
    pieces = U.split_node(nodes[0], tok, 40)
    assert all(tok.count(p.payload) <= 40 for p in pieces)
    assert all(tok.count(p.payload) >= 40 - 5 for p in pieces[:-1])


def test_split_copy_and_short_nodes_untouched(tok):
    rng = random.Random(9)
    src, nodes = build([C("-" * 4000), T("Short sentence."), T(para(rng, 30))])
    nodes[0].payload = "x " * 3000                                 # never split a copy node
    out = U.split_long_nodes(nodes, tok, 100)
    assert out[0] is nodes[0] and out[1] is nodes[1] and len(out) > 3


def test_split_per_kind_caps_with_config(tok):
    rng = random.Random(4)
    cell = " ".join(rng.choice(WORDS) for _ in range(600)) + "."
    sig = " ".join(rng.choice(WORDS) for _ in range(200)).upper()
    text = para(rng, 40)
    src, nodes = build([T(cell, kind="table_cell", table={"group": "t1", "row": 0, "col": 0}),
                        T(sig, kind="signature"), T(text)])
    out = U.split_long_nodes(nodes, tok, CFG)
    by_root = {}
    for p in out:
        by_root.setdefault(U.root_node_id(p.node_id), []).append(p)
    assert all(tok.count(p.payload) <= CFG.table_cap for p in by_root["n00000"]) and len(by_root["n00000"]) > 1
    assert all(tok.count(p.payload) <= CFG.signature_cap for p in by_root["n00001"])
    assert all(tok.count(p.payload) <= CFG.source_cap for p in by_root["n00002"])
    assert all(p.table == {"group": "t1", "row": 0, "col": 0} for p in by_root["n00000"])
    units = U.pack_units(out, "Cat/6", tok, CFG, "SEG")
    check_units(units, out, src, tok, "SEG", allow_single=False)


def test_unwrap_rules():
    assert U.unwrap("one line\n  continues   here\n\n  after break") == "one line continues here after break"
    assert U.unwrap("REGULA-\n  TION apply") == "REGULA-TION apply"
    assert U.unwrap("1998 -\n  next") == "1998 - next"


# --------------------------------------------------------------------------------- parse_output
@pytest.fixture()
def unit3(tok):
    _, nodes = build([T("First."), T("Second."), T("Third.")])
    (u,) = U.pack_units(nodes, "Cat/7", tok, CFG, "SEG")
    assert len(u.markers) == 3
    return u


def test_parse_ok_and_tolerant(unit3):
    m0, m1, m2 = unit3.markers
    n0, n1, n2 = unit3.node_ids
    want = {n0: "Một.", n1: "Hai.", n2: "Ba."}
    r = U.parse_output(unit3, f"{m0} Một.\n{m1} Hai.\n{m2} Ba.")
    assert r.hard == [] and r.segments == want
    r = U.parse_output(unit3, f"\n\n{m0}   Một.  \n\n\n{m1}\nHai.\n\n  {m2}Ba. \n\n")
    assert r.hard == [] and r.segments == want


def test_parse_hard_failures(unit3):
    m0, m1, m2 = unit3.markers
    n0, n1, n2 = unit3.node_ids
    assert U.parse_output(unit3, "").hard == ["empty_output"]
    assert U.parse_output(unit3, "  \n ").hard == ["empty_output"]
    r = U.parse_output(unit3, f"{m0} Một.\n{m2} Ba.")
    assert r.hard == [f"marker_missing:{n1}"] and r.segments == {n0: "Một.", n2: "Ba."}
    r = U.parse_output(unit3, f"{m0} Một.\n{m1} Hai.\n{m1} Hai lại.\n{m2} Ba.")
    assert r.hard == [f"marker_duplicate:{n1}"] and r.segments[n1] == "Hai."
    r = U.parse_output(unit3, f"{m0} Một.\n{m2} Ba.\n{m1} Hai.")
    assert r.hard == ["marker_order"] and r.segments == {n0: "Một.", n1: "Hai.", n2: "Ba."}
    other = U.marker("SEG", 999)
    r = U.parse_output(unit3, f"{m0} Một.\n{other} Lạ.\n{m1} Hai.\n{m2} Ba.")
    assert r.hard == [f"marker_unexpected:{other}"] and r.segments[n0] == "Một."
    assert r.extra["dropped"] == ["Lạ."]
    r = U.parse_output(unit3, f"Lời dẫn.\n{m0} Một.\n{m1} Hai.\n{m2} Ba.")
    assert r.hard == ["unassigned_text"]
    r = U.parse_output(unit3, f"{m0} Một.\n{m1}\n\n{m2} Ba.")
    assert r.hard == [f"segment_empty:{n1}"] and n1 not in r.segments
    r = U.parse_output(unit3, "Không có marker nào.")
    assert set(r.hard) == {f"marker_missing:{n}" for n in unit3.node_ids} | {"unassigned_text"}


def test_parse_marker_remnants_removed(unit3):
    m0, m1, m2 = unit3.markers
    n0 = unit3.node_ids[0]
    r = U.parse_output(unit3, f"{m0} Một. [SEG_12]\n{m1} Hai.\n{m2} Ba.")
    assert r.hard == ["marker_unexpected:[SEG_12]"] and r.segments[n0] == "Một."


def test_parse_foreign_namespace_kept(tok):
    src_text = "Source keeps its own marker [SEG_000001] in the text."
    ns = U.marker_namespace(src_text)
    assert ns != "SEG"
    _, nodes = build([T(src_text), T("Second node.")])
    (u,) = U.pack_units(nodes, "Cat/8", tok, CFG, ns)
    m0, m1 = u.markers
    r = U.parse_output(u, f"{m0} Nguồn giữ marker [SEG_000001] trong văn bản.\n{m1} Nút hai.")
    assert r.hard == []
    assert r.segments[u.node_ids[0]] == "Nguồn giữ marker [SEG_000001] trong văn bản."


def test_parse_single_node_unit(tok):
    _, nodes = build([T("Only one node here.")])
    (u,) = U.pack_units(nodes, "Cat/9", tok, CFG, "SEG")
    assert u.markers == [] and u.payload == nodes[0].payload
    r = U.parse_output(u, "  \n Chỉ một nút. \n")
    assert r.hard == [] and r.segments == {u.node_ids[0]: "Chỉ một nút."}
    assert U.parse_output(u, " \n").hard == ["empty_output"]


# ---------------------------------------------------------------------------------------- budget
@pytest.mark.parametrize("src_tok,prompt_tok,want", [
    (100, 1000, 378),            # ceil(250) + 128
    (101, 1000, 381),            # ceil(252.5) = 253
    (10, 500, 256),              # floor 256
    (0, 100, 256),
    (768, 900, 2048),
    (768, 3500, 4096 - 3500 - 32),   # remaining binds
])
def test_output_budget(src_tok, prompt_tok, want):
    import math
    got = U.output_budget(src_tok, prompt_tok, CFG)
    assert got == want
    assert got == min(CFG.max_sequence - prompt_tok - CFG.margin, max(256, math.ceil(2.5 * src_tok) + 128))
    assert U.budget_fits(src_tok, prompt_tok, CFG) == (got == U.target_budget(src_tok))


# -------------------------------------------------------------------------------------- recovery
def recovery_doc(tok):
    rng = random.Random(21)
    specs = [T(wrap(para(rng, 1, 5, 8))), C("-----"), T(wrap(para(rng, 20, 14, 18))),
             T(wrap(para(rng, 2, 6, 10))), T(wrap(para(rng, 3, 8, 12))), T("Short line.")]
    src, nodes = build(specs)
    units = U.pack_units(nodes, "Cat/10", tok, CFG, "SEG")
    assert len(units) == 1 and len(units[0].node_ids) == 5
    assert any(tok.count(n.payload) > CFG.recovery_cap for n in nodes if n.translatable)
    return src, nodes, units[0]


def test_recovery_independent(tok):
    src, nodes, unit = recovery_doc(tok)
    by_id = {n.node_id: n for n in nodes}
    kids = U.recovery_units(unit, by_id, tok, CFG, "SEG", independent=True)
    assert all(len(k.node_ids) == 1 for k in kids)
    ordered_roots = list(dict.fromkeys(U.root_node_id(k.node_ids[0]) for k in kids))
    assert ordered_roots == list(unit.node_ids)
    assert [k.unit_id for k in kids] == [f"{unit.unit_id}.r{i}" for i in range(1, len(kids) + 1)]
    for k in kids:
        n = by_id[k.node_ids[0]]
        assert k.kind == "single" and k.markers == [] and k.payload == n.payload
        assert k.split_from == unit.unit_id and k.scope_id == unit.scope_id
        assert k.source_tokens == tok.count(k.payload) <= CFG.recovery_cap
        assert k.raw_sha256 == sha(src[k.raw_start:k.raw_end])
    assert len(by_id) > len(nodes)                                  # long parent split into registered pieces
    for nid in unit.node_ids:
        pieces = [by_id[k.node_ids[0]] for k in kids if U.root_node_id(k.node_ids[0]) == nid]
        assert "".join(p.raw for p in pieces) == by_id[nid].raw


def test_recovery_regroup(tok):
    src, nodes, unit = recovery_doc(tok)
    by_id = {n.node_id: n for n in nodes}
    first_marker = dict(zip(unit.node_ids, unit.markers))
    kids = U.recovery_units(unit, by_id, tok, CFG, "SEG", independent=False)
    assert len(kids) >= 2
    assert [k.unit_id for k in kids] == [f"{unit.unit_id}.r{i}" for i in range(1, len(kids) + 1)]
    flat = [nid for k in kids for nid in k.node_ids]
    assert [x for x in dict.fromkeys(U.root_node_id(x) for x in flat)] == unit.node_ids
    assert len(set(flat)) == len(flat)
    assert any("." in x for x in flat)                               # the long node was split
    pos = {nid: i for i, nid in enumerate(by_id)}
    for k in kids:
        assert k.kind == "recovery" and k.split_from == unit.unit_id
        assert k.source_tokens == tok.count(k.payload) <= CFG.recovery_cap
        assert all(x in by_id for x in k.node_ids)                    # new pieces registered
        kn = [by_id[x] for x in k.node_ids]
        assert k.payload == U.render_payload(kn, k.markers)
        if len(kn) > 1:
            assert k.markers == [U.marker("SEG", pos[x]) for x in k.node_ids]
            assert len(set(k.markers)) == len(k.markers)
            for x, m in zip(k.node_ids, k.markers):
                if x in first_marker:
                    assert m == first_marker[x]                       # same marker as first pass
        assert k.raw_sha256 == sha(src[k.raw_start:k.raw_end])
    # pieces of a split node partition it
    for nid in unit.node_ids:
        parts = [by_id[x] for x in flat if U.root_node_id(x) == nid]
        assert "".join(p.raw for p in parts) == by_id[nid].raw
    # union of children -> parent segments
    segs = {x: f"t({x})" for x in flat}
    merged = U.merge_child_segments(unit, kids, segs)
    assert set(merged) == set(unit.node_ids)
    long_id = next(nid for nid in unit.node_ids if f"{nid}.1" in flat)
    assert merged[long_id] == " ".join(f"t({x})" for x in flat if U.root_node_id(x) == long_id)
    segs.pop(flat[-1])
    assert U.root_node_id(flat[-1]) not in U.merge_child_segments(unit, kids, segs)


def test_recovery_small_unit_is_not_resent_identically(tok):
    src, nodes = build([T("First short clause."), T("Second short clause."), T("Third one.")])
    (unit,) = U.pack_units(nodes, "Cat/11", tok, CFG, "SEG")
    by_id = {n.node_id: n for n in nodes}
    kids = U.recovery_units(unit, by_id, tok, CFG, "SEG", independent=False)
    assert len(kids) == 2 and [x for k in kids for x in k.node_ids] == unit.node_ids
    assert all(k.payload != unit.payload for k in kids)


def test_recovery_unknown_node_raises(tok):
    _, nodes = build([T("One."), T("Two.")])
    (unit,) = U.pack_units(nodes, "Cat/12", tok, CFG, "SEG")
    with pytest.raises(ValueError):
        U.recovery_units(unit, {nodes[0].node_id: nodes[0]}, tok, CFG, "SEG", independent=True)


# ------------------------------------------------------------------------- marker cost (measured)
def test_marker_token_cost(tok, smoke_nodes):
    """Markers cost 7 tokens each ('[','SE','G','_','000','123',']') + the newline joiner."""
    assert tok.count(U.marker("SEG", 123)) == 7
    per_node, eight = [], []
    for row, src, nodes in smoke_nodes:
        split = U.split_long_nodes(nodes, tok, CFG)
        by_id = {n.node_id: n for n in split}
        for u in U.pack_units(split, row["sample_id"], tok, CFG, U.marker_namespace(src)):
            if len(u.node_ids) < 2:
                continue
            plain = sum(tok.count(by_id[x].payload) for x in u.node_ids)
            overhead = u.source_tokens - plain
            per_node.append(overhead / len(u.node_ids))
            if len(u.node_ids) == 8:
                eight.append(overhead)
    assert per_node and eight
    med8 = statistics.median(eight)
    print(f"\nmarker overhead: per node median {statistics.median(per_node):.2f} "
          f"(min {min(per_node):.2f}, max {max(per_node):.2f}); 8-node units n={len(eight)} "
          f"median {med8} min {min(eight)} max {max(eight)} tokens")
    assert 6 <= statistics.median(per_node) <= 9 and 50 <= med8 <= 72


# ------------------------------------------------------------------- integration with parse.py
def test_integration_parse_smoke6(tok, smoke6):
    try:
        from docmt.steps.s01_parse import parse_document
    except Exception as e:                                   # missing or not importable yet
        pytest.skip(f"docmt.steps.s01_parse not available: {e!r}")
    for row, src in smoke6:
        try:
            nodes = parse_document(src, row["sample_id"], row["src_lang"], tok)
        except Exception as e:
            pytest.skip(f"parse_document raised on {row['sample_id']}: {e!r}")
        if "".join(n.raw for n in nodes) != src:
            pytest.skip(f"parse invariants not met on {row['sample_id']}")
        split = U.split_long_nodes(nodes, tok, CFG)
        assert "".join(n.raw for n in split) == src
        ns = U.marker_namespace(src)
        units = U.pack_units(split, row["sample_id"], tok, CFG, ns)
        check_units(units, split, src, tok, ns, allow_single=False)

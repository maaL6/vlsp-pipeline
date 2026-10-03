import json
from pathlib import Path

import pytest

from docmt.backends.mock import MockBackend
from docmt.core.config import RunConfig
from docmt.context.memory import Memory
from docmt.steps.s01_parse import parse_document, reconstruct
from docmt.engine.api import translate_document


SOURCE = "Article 24. Duties\r\n\r\n1. The agency shall submit the report.\r\n\r\n2. The applicant shall pay 300,000 VND.\r\n\r\nArticle 26. Reward\r\n\r\nThe agency shall reward the applicant.\r\n"


def execute(tmp_path, tok, rules=(), **overrides):
    source = tmp_path / "source.md"
    source.write_bytes(SOURCE.encode())
    cfg = RunConfig(backend="mock", run_id="gate_test", **overrides)
    backend = MockBackend(rules=rules, tok=tok)
    result = translate_document(source, tmp_path / "prediction.md", "en", "vi", cfg,
                                backend=backend, tok=tok, log_dir=tmp_path / "logs", doc_id="Cat/1")
    traces = [json.loads(x) for x in (tmp_path / "logs/generation_trace.jsonl").read_text().splitlines()]
    return result, traces, backend


def test_crlf_roundtrip_and_source_jump(tok):
    nodes = parse_document(SOURCE, "Cat/1", "en", tok)
    assert reconstruct(nodes) == SOURCE
    headings = [n.protected["heading_number"] for n in nodes if n.protected.get("heading_type") == "article"]
    assert headings == ["24", "26"]


def test_numeric_pipe_cells_are_copied(tok):
    source = "| Item | Amount |\n|---|---|\n| Fee | 300,000 |\n"
    nodes = parse_document(source, "Cat/1", "en", tok)
    assert reconstruct(nodes) == source
    numeric = [n for n in nodes if n.content_raw.strip() == "300,000"]
    assert len(numeric) == 1 and not numeric[0].translatable


@pytest.mark.parametrize("fault", ["drop_marker", "duplicate_marker", "swap_markers", "text_before_marker", "no_eos"])
def test_single_recovery_recovers_first_pass(tmp_path, tok, fault):
    result, trace, backend = execute(tmp_path, tok, rules=[dict(fault=fault, times=1, match={"min_markers": 2})])
    assert result.status == "completed", result.diagnostics
    assert result.diagnostics["recovery_requests"] > 0
    assert max(x["attempt"] for x in trace) == 1
    assert not result.diagnostics["uncovered_nodes"]
    assert "[SEG_" not in (tmp_path / "prediction.md").read_text()
    recovery = [x for x in trace if x["attempt"] == 1]
    assert all(x["source_tokens"] <= 256 for x in recovery)
    assert all(not any(e["kind"] == "history" for e in x["evidence"]) for x in recovery)
    if fault != "no_eos":
        assert all(len(x["node_ids"]) == 1 for x in recovery)


def test_persistent_failure_is_unresolved(tmp_path, tok):
    result, trace, _ = execute(tmp_path, tok, rules=[dict(fault="no_eos")])
    assert result.status == "completed_with_unresolved"
    assert result.diagnostics["unresolved_units"]
    assert max(x["attempt"] for x in trace) == 1


def test_empty_persistent_output_is_failed(tmp_path, tok):
    result, trace, _ = execute(tmp_path, tok, rules=[dict(fault="empty")])
    assert result.status == "failed"
    assert not (tmp_path / "prediction.md").exists()


def test_memory_isolates_doc_scope_and_rejects_unresolved():
    memory = Memory()
    assert not memory.add("a", "main", "bad", "s", "gold-or-unresolved", False)
    assert memory.add("a", "main", "good", "s", "accepted", True)
    assert not memory.recent("b", "main", 2)
    assert not memory.recent("a", "appendix_1", 2)
    assert [x["unit_id"] for x in memory.recent("a", "main", 2)] == ["good"]


def test_history_sequential_and_recovery_excludes_history(tmp_path, tok):
    result, trace, backend = execute(tmp_path, tok, target_history=True, source_context=True,
                                     max_nodes=2, rules=[dict(fault="drop_marker", times=1, match={"min_markers": 2})])
    assert result.status == "completed"
    assert max(backend.max_inflight_by_doc.values()) == 1
    assert all(not any(e["kind"] == "history" for e in x["evidence"]) for x in trace if x["attempt"] == 1)
    assert all(x["input_tokens"] + x["max_new_tokens"] + 32 <= 4096 for x in trace)


def test_runtime_splits_table_and_signature_by_their_caps(tmp_path, tok):
    from docmt.core.datatypes import Node
    source = "legal " * 600 + "\n" + "MINISTER " * 400
    input_path = tmp_path / "source.md"
    input_path.write_text(source)
    pivot = source.index("\n") + 1
    def parse(src, *args):
        return [Node("n0", "table_cell", 0, pivot, "", src[:pivot-1], "\n", src[:pivot-1],
                     table=dict(group="t1", row=0, col=0, certain=True)),
                Node("n1", "signature", pivot, len(src), "", src[pivot:], "", src[pivot:])]
    cfg = RunConfig(backend="mock", run_id="caps")
    backend = MockBackend(tok=tok)
    result = translate_document(input_path, tmp_path / "prediction.md", "en", "vi", cfg,
                                backend=backend, tok=tok, components=dict(parse_document=parse),
                                log_dir=tmp_path / "logs")
    assert result.status == "completed", result.diagnostics
    traces = [json.loads(x) for x in (tmp_path / "logs/generation_trace.jsonl").read_text().splitlines()]
    assert traces
    for row in traces:
        if row["attempt"] == 1:
            assert row["source_tokens"] <= 256
        elif row["source_offsets"][0][0] < pivot:
            assert row["source_tokens"] <= 384
        else:
            assert row["source_tokens"] <= 128


def test_source_only_context_follows_source_order(tmp_path, tok):
    result, trace, backend = execute(tmp_path, tok, source_context=True, max_nodes=1)
    assert result.status == "completed"
    assert max(backend.max_inflight_by_doc.values()) == 1
    first_pass = [r for r in trace if r["attempt"] == 0]
    starts = [r["source_offsets"][0][0] for r in first_pass]
    assert starts == sorted(starts)


def test_marker_recovery_splits_real_369_token_source(tmp_path, tok):
    from docmt.core.datatypes import Node
    from docmt.steps.s03_pack import pack_units
    from docmt.steps.s06_parse_output import merge_child_segments
    from docmt.steps.s08_recovery import recovery_units
    payload = (Path(__file__).parent / "fixtures/recovery_369_source.txt").read_text()
    assert tok.count(payload) == 369
    suffix = "\n\nThe agency shall submit a report."
    source = payload + suffix
    parents = [Node("n0", "paragraph", 0, len(payload), "", payload, "", payload),
               Node("n1", "paragraph", len(payload), len(source), "\n\n", suffix[2:], "", suffix[2:])]
    cfg = RunConfig(backend="mock", run_id="cap_regression")
    unit, = pack_units(parents, "Fixture/1", tok, cfg, "SEG")
    by_id = {n.node_id:n for n in parents}
    children = recovery_units(unit, by_id, tok, cfg, "SEG", independent=True)
    assert len(children) > len(parents)
    assert all(child.source_tokens <= 256 and len(child.node_ids) == 1 and not child.markers for child in children)
    pieces = [by_id[nid] for child in children for nid in child.node_ids if nid.startswith("n0.")]
    assert "".join(p.raw for p in pieces) == payload
    assert all(a.raw_end == b.raw_start for a,b in zip(pieces,pieces[1:]))
    merged = merge_child_segments(unit, children, {c.node_ids[0]:"translated" for c in children})
    assert set(merged) == {"n0", "n1"}
    path = tmp_path / "source.md"
    path.write_text(source)
    backend = MockBackend(tok=tok, rules=[dict(fault="drop_marker", times=1, match={"min_markers":2})])
    result = translate_document(path,tmp_path/"prediction.md","en","vi",cfg,backend=backend,tok=tok,
                                components=dict(parse_document=lambda *args:parents),log_dir=tmp_path/"logs")
    assert result.status == "completed", result.diagnostics
    traces = [json.loads(l) for l in (tmp_path/"logs/generation_trace.jsonl").read_text().splitlines()]
    recovered = [r for r in traces if r["attempt"] == 1]
    assert recovered and all(r["source_tokens"] <= 256 for r in recovered)
    assert max(r["attempt"] for r in traces) == 1

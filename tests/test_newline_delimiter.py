"""payload_delimiter "newline" (phương án B): units packed as with markers, nodes joined by '\\n'
only, output split back by line; a line-count mismatch is a marker-type failure (s08)."""
from pathlib import Path
from types import SimpleNamespace

from docmt.core.datatypes import Unit
import pytest

from docmt.steps.s03_pack import render_payload, unit_markers, use_markers
from docmt.steps.s06_parse_output import parse_output
from docmt.steps.s08_recovery import is_marker_failure, recovery_mode

V1 = Path(__file__).resolve().parents[1]


def _unit(n):
    return Unit(unit_id="d:u0000", doc_id="d", scope_id="main", kind="text",
                node_ids=[f"n{i:05d}" for i in range(n)], markers=[], payload="x",
                source_tokens=1, raw_start=0, raw_end=1, raw_sha256="")


def test_use_markers_flag():
    assert use_markers(SimpleNamespace(extra={}))
    assert use_markers(SimpleNamespace(extra={"payload_delimiter": "marker"}))
    assert not use_markers(SimpleNamespace(extra={"payload_delimiter": "newline"}))
    with pytest.raises(ValueError):
        use_markers(SimpleNamespace(extra={"payload_delimiter": "lines"}))


def test_unit_markers():
    marker_cfg, newline_cfg = SimpleNamespace(extra={}), SimpleNamespace(extra={"payload_delimiter": "newline"})
    assert unit_markers("SEG", [3, 4], marker_cfg) == ["[SEG_000003]", "[SEG_000004]"]
    assert unit_markers("SEG", [3], marker_cfg) == []          # a single node is always bare
    assert unit_markers("SEG", [3, 4], newline_cfg) == []


def test_render_payload_without_markers():
    nodes = [SimpleNamespace(payload=p) for p in ("Article 1.", "Scope.")]
    assert render_payload(nodes, []) == "Article 1.\nScope."
    assert render_payload(nodes[:1], []) == "Article 1."


def test_parse_lines():
    ok = parse_output(_unit(3), "Điều 1.\n\n  Phạm vi. \nĐối tượng.\n")
    assert ok.hard == [] and ok.segments == {"n00000": "Điều 1.", "n00001": "Phạm vi.", "n00002": "Đối tượng."}
    bad = parse_output(_unit(3), "Điều 1. Phạm vi.\nĐối tượng.")
    assert bad.hard == ["line_count_mismatch:2/3"] and bad.segments == {}
    assert is_marker_failure(bad.hard)
    assert parse_output(_unit(2), "  ").hard == ["empty_output"]


def test_recovery_mode_newline_units_always_independent():
    newline_unit = _unit(3)                                        # markers == [] -> newline mode
    marker_unit = _unit(3)
    marker_unit.markers = ["[SEG_000000]", "[SEG_000001]", "[SEG_000002]"]
    heading = ["protected_heading_lost:n00001"]
    assert recovery_mode(newline_unit, heading) == ("independent", True)
    assert recovery_mode(newline_unit, ["line_count_mismatch:2/3"]) == ("independent", True)
    assert recovery_mode(marker_unit, heading) == ("regroup", False)          # A unchanged
    assert recovery_mode(marker_unit, ["marker_missing:n00001"]) == ("independent", True)
    assert recovery_mode(_unit(1), ["no_eos"]) == ("regroup", False)


def test_newline_config_run_dir_mock(tmp_path):
    from conftest import INPUTS, TOKENIZER, read_jsonl
    from docmt.cli import main
    row = [r for r in read_jsonl(INPUTS / "smoke6_doc_direction_manifest.jsonl") if r["direction"] == "en-vi"][0]
    src = tmp_path / "in" / (row["sample_id"] + ".md")
    src.parent.mkdir(parents=True)
    src.write_bytes((INPUTS / row["source_path"]).read_bytes())
    work = tmp_path / "work"
    rc = main(["run-dir", "--input-dir", str(tmp_path / "in"), "--output-dir", str(tmp_path / "out"),
               "--work-dir", str(work), "--direction", "en-vi", "--config",
               str(V1 / "configs/hy_a1_structure_newline.json"), "--run-id", "t_newline",
               "--backend", "mock", "--tokenizer", str(TOKENIZER)])
    assert rc == 0
    prompts = (work / "logs/prompts.jsonl").read_text(encoding="utf-8")
    assert "[SEG" not in prompts
    import json
    rows = [json.loads(x) for x in (work / "logs/generation_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    grouped = [r for r in rows if r["attempt"] == 0 and len(r["node_ids"]) > 1]
    assert grouped and all(r["markers"] == [] for r in grouped)       # still grouped, never marked

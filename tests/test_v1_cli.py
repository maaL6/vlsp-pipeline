"""Folder I/O (run-dir), inspect --stop-after, preset JSON, replay backend."""
import json
import shutil
from pathlib import Path

import pytest

from conftest import INPUTS, TOKENIZER, read_jsonl

V1 = Path(__file__).resolve().parents[1]


@pytest.fixture()
def folder(tmp_path):
    """Two smoke6 sources copied into a nested input folder (category/id.md)."""
    rows = [r for r in read_jsonl(INPUTS / "smoke6_doc_direction_manifest.jsonl") if r["direction"] == "en-vi"][:2]
    src = tmp_path / "in"
    for r in rows:
        dst = src / (r["sample_id"] + ".md")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(INPUTS / r["source_path"], dst)
    return src, [r["sample_id"] for r in rows]


def test_folder_rows(folder):
    from docmt.cli import folder_rows
    src, ids = folder
    rows = folder_rows(src, "en-vi")
    assert [r["sample_id"] for r in rows] == sorted(ids)
    assert all(r["direction"] == "en-vi" and r["source_path"].endswith(".md") for r in rows)


def test_run_dir_mock_writes_same_tree(folder, tmp_path):
    from docmt.cli import main
    src, ids = folder
    out = tmp_path / "out"
    rc = main(["run-dir", "--input-dir", str(src), "--output-dir", str(out), "--direction", "en-vi",
               "--config", str(V1 / "configs/hy_mt_vlsp26.json"), "--run-id", "t_run_dir",
               "--backend", "mock", "--tokenizer", str(TOKENIZER)])
    assert rc == 0
    written = sorted(p.relative_to(out).with_suffix("").as_posix() for p in out.rglob("*.md"))
    assert written == sorted(ids)
    for sid in ids:
        text = (out / (sid + ".md")).read_text(encoding="utf-8")
        assert text.strip() and "[SEG_" not in text
    work = Path(str(out) + ".work")
    assert (work / "logs" / "generation_trace.jsonl").exists()
    assert not list(out.rglob("*.jsonl")), "logs must stay out of the output folder"


def test_inspect_stages_are_consistent(tmp_path):
    from docmt.core.config import make_config
    from docmt.core.tok import HyTok
    from docmt.engine.inspect import inspect_items
    from docmt.inputs.sources import expand_rows
    items = expand_rows(read_jsonl(INPUTS / "smoke6_doc_direction_manifest.jsonl"), INPUTS)[:2]
    cfg, tok = make_config("hy_mt_vlsp26", run_id="t_inspect"), HyTok(TOKENIZER)
    stats = {s: inspect_items(items, cfg, tok, s, tmp_path / f"{s}.jsonl") for s in ("parse", "split", "pack", "prompt")}
    assert all(not s["failed"] for s in stats.values())
    assert stats["pack"]["records"] == stats["prompt"]["records"] > 0
    assert stats["split"]["records"] >= stats["parse"]["records"]
    prompts = read_jsonl(tmp_path / "prompt.jsonl")
    assert all(p["fits"] and p["total_tokens"] <= cfg.max_sequence for p in prompts)


def test_default_preset_loads():
    from docmt.core.config import RunConfig
    data = json.loads((V1 / "configs/hy_mt_vlsp26.json").read_text(encoding="utf-8"))
    cfg = RunConfig().apply(data)
    assert cfg.backend == "hf" and cfg.batch_size == 8 and cfg.repetition_penalty == 1.0
    assert not (cfg.do_sample or cfg.source_context or cfg.soft_glossary or cfg.target_history)
    assert cfg.extra["payload_delimiter"] == "newline" and cfg.extra["legal_prose_blocks"] is True


def test_replay_backend_hit_and_miss(tmp_path):
    import hashlib
    from docmt.backends.replay import ReplayBackend
    from docmt.core.datatypes import GenRequest
    trace = tmp_path / "t.jsonl"
    trace.write_text(json.dumps({"prompt_sha256": hashlib.sha256(b"P1").hexdigest(), "output": "out1", "eos": False,
                                 "stop_type": "limit", "input_tokens": 3, "output_tokens": 9}) + "\n", encoding="utf-8")
    be = ReplayBackend([trace])
    r1, r2 = be.generate([GenRequest(request_id="a", prompt="P1", max_new_tokens=9, meta={}),
                          GenRequest(request_id="b", prompt="P2", max_new_tokens=9, meta={"payload": "src"})])
    assert (r1.text, r1.eos, r1.stop_type, r1.output_tokens) == ("out1", False, "limit", 9)
    assert (r2.text, r2.eos) == ("src", True) and be.misses == ["b"] and be.stats["replay_miss"] == 1

"""CPU checks for request ordering, recovery, cache and glossary direction."""
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pytest
from docmt.core.config import RunConfig
from docmt.core.datatypes import GenRequest, GenResult, Unit, Node
from docmt.engine.run import _Run
from docmt.engine.scheduler import _Runner
from docmt.tracing.cache import ResumeCache
from docmt.context.glossary import Glossary
from docmt.context.evidence import select_evidence
from docmt.steps.s04_prompt import render_unit_prompt


class Tok:
    def count(self, text):
        return len(text)


class BatchBackend:
    batching = True
    name = "test"

    def __init__(self):
        self.batches = []

    def decode_config(self):
        return {}

    def generate(self, reqs):
        self.batches.append([r.request_id for r in reqs])
        return [GenResult(r.request_id, r.prompt, True, "eos", len(r.prompt), 1)
                for r in reversed(reqs)]  # scheduler must route by id


class Job:
    def __init__(self, requests, recovery=()):
        self.requests, self.recovery = requests, recovery
        self.results = []

    def flow(self):
        self.results.extend((yield self.requests))
        if self.recovery:
            self.results.extend((yield list(self.recovery)))

    def fail(self, exc):
        raise AssertionError("job failed") from exc


def request(name, length, attempt=0):
    return GenRequest(name, name[0]*length, 256+length,
                      meta={"doc_id": name[0], "attempt": attempt})


def execute(policy, jobs, limit=None, cached=None):
    cfg = RunConfig(batch_size=2, extra={"batch_policy": policy})
    backend = BatchBackend()
    errors = []
    cache = ResumeCache(None, cfg.sha256())
    run = _Run(cfg, Tok(), backend, None, SimpleNamespace(error=errors.append), cache, "test")
    if cached:
        cache.put(run.cache_key(cached), GenResult(cached.request_id, cached.prompt, True, "eos", len(cached.prompt), 1))
    key = run._length_key if policy == "length" else None
    run._schedule([_Runner(job, limit, key) for job in jobs])
    assert not errors
    return run, backend


def test_length_batches_preserve_results_and_recovery_order():
    reqs = [request("a0", 180), request("a1", 10), request("a2", 90), request("a3", 20)]
    retry = request("a0.r1", 5, 1)
    for policy in ("fifo", "length"):
        job = Job(reqs, [retry])
        run, backend = execute(policy, [job])
        assert [r.request_id for r in job.results] == [r.request_id for r in reqs]+[retry.request_id]
        assert [r.text for r in job.results] == [r.prompt for r in reqs]+[retry.prompt]
        assert run.totals["recovery_requests"] == 1
        assert backend.batches[:2] == ([["a1", "a3"], ["a2", "a0"]] if policy == "length"
                                      else [["a0", "a1"], ["a2", "a3"]])


def test_length_batch_cache_hits_do_not_stall():
    reqs = [request("a0", 100), request("a1", 10), request("a2", 50)]
    job = Job(reqs)
    run, backend = execute("length", [job], cached=reqs[1])
    assert run.totals["cache_hits"] == 1
    assert run.totals["requests_sent"] == 2
    assert [r.request_id for r in job.results] == [r.request_id for r in reqs]
    assert backend.batches == [["a2", "a0"]]


def test_length_batch_respects_per_document_inflight_limit():
    jobs = [Job([request("a0", 100), request("a1", 10)]),
            Job([request("b0", 90), request("b1", 20)])]
    _, backend = execute("length", jobs, limit=1)
    assert all(len({rid[0] for rid in batch}) == len(batch) for batch in backend.batches)
    assert all(len(job.results) == 2 for job in jobs)


@pytest.mark.parametrize("lang,text,want", [("en", "Article", "Article translates to Điều"),
                                          ("vi", "Điều", "Điều translates to Article")])
def test_glossary_only_uses_explicit_source_language_and_native_terms(lang, text, want):
    cfg = RunConfig(soft_glossary=True, source_context=False)
    glossary = Glossary([{"en": "Article", "vi": "Điều"}])
    node = Node("n0", "paragraph", 0, len(text), "", text, "", text)
    unit = Unit("a:u0", "a", "main", "text", ["n0"], [], text, len(text), 0, len(text), "")
    items = select_evidence(unit, {"n0": node}, None, None, glossary, Tok(), cfg, "a", src_lang=lang)
    assert len(items) == 1 and items[0].kind == "glossary"
    prompt, template = render_unit_prompt(unit, items, lang, "vi" if lang == "en" else "en", cfg, None)
    assert want in prompt and template == "default:HY_TERMS"
    assert "[Background Information]" not in prompt


@pytest.mark.parametrize("variant", ["batch_length", "glossary"])
@pytest.mark.parametrize("direction", ["en-vi", "vi-en"])
def test_real_pipeline_config_recovery_and_resume(tmp_path, variant, direction):
    import json
    from tokenizers import Tokenizer, models, pre_tokenizers
    from docmt.core.tok import HyTok
    from docmt.backends.mock import MockBackend
    from docmt.engine.api import run_manifest

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokpath = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokpath))
    tok = HyTok(tokpath)
    cfg = RunConfig().apply(json.loads((ROOT / f"configs/hy_a1_v13_{variant}.json").read_text()))
    cfg.apply({"run_id": f"test_{variant}_{direction}", "backend": "mock", "tokenizer_path": str(tokpath),
               "glossary_path": str(ROOT / "configs/legal_glossary_v13.jsonl")})
    text = "# Article\n\nThe administrative violations shall be reviewed.\n\nThe court has jurisdiction.\n" if direction == "en-vi" else \
           "# Điều\n\nCác vi phạm hành chính phải được xem xét.\n\nTòa án có thẩm quyền.\n"
    rows = []
    for i in range(3):
        path = tmp_path / f"doc{i}.md"
        path.write_text(text + "\n" + ("Article" if direction == "en-vi" else "Điều") * (i + 1), encoding="utf-8")
        rows.append({"sample_id": f"legal/doc{i}", "source_path": path.name, "direction": direction})
    backend = MockBackend(tok=tok, batching=True, rules=[{"fault": "empty", "times": 1}])
    out = tmp_path / "run"
    summary = run_manifest(rows, tmp_path, out, cfg, backend, tok, directions=direction)
    assert summary[direction]["documents"] == 3
    assert not summary[direction]["statuses"].get("failed")
    assert summary[direction]["recovery_requests"] > 0
    assert len(list((out / "predictions" / direction).rglob("*.md"))) == 3
    assert all(len(batch) <= 4 for batch in backend.batches)
    calls = backend.calls
    if variant == "glossary":
        want = "administrative violations translates to vi phạm hành chính" if direction == "en-vi" else \
               "vi phạm hành chính translates to administrative violations"
        assert any(want in call["prompt"] for call in calls)
        assert all("[Background Information]" not in call["prompt"] for call in calls)
    else:
        assert all("translates to" not in call["prompt"] for call in calls)
    before = {p.relative_to(out): p.read_bytes() for p in (out / "predictions").rglob("*.md")}
    replay = MockBackend(tok=tok, batching=True, rules=[{"fault": "empty", "times": 1}])
    run_manifest(rows, tmp_path, out, cfg, replay, tok, directions=direction)
    assert not replay.calls
    assert before == {p.relative_to(out): p.read_bytes() for p in (out / "predictions").rglob("*.md")}

"""Exercise all 50 public docs x 2 directions with a fake model, never report BLEU."""
import json
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from docmt.core.config import RunConfig
from docmt.core.tok import HyTok
from docmt.backends.mock import MockBackend
from docmt.engine.api import run_manifest
from docmt.cli import folder_rows


def main():
    build = ROOT / "kaggle/public_test/build"
    bench = build / "original/benchmark"
    csv.field_size_limit(2**31 - 1)
    with (bench / "data/en-vi-public_test.csv").open(encoding="utf-8", newline="") as f:
        docs = [{"en": r["english_markdown"], "vi": r["vietnamese_markdown"],
                 "en_path": r["english_path"], "vi_path": r["vietnamese_path"]} for r in csv.DictReader(f)]
    sources = build / "preflight_sources"
    for doc in docs:
        for lang in ("en", "vi"):
            p = sources / doc[f"{lang}_path"]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(doc[lang], encoding="utf-8", newline="")
    tok = HyTok(ROOT / "hy-mt2-1.8b-trained/tokenizer.json")
    report = {"model": "mock echo; NOT GPU benchmark or translations", "runs": {}}
    for variant in ("batch_length", "glossary"):
        for direction in ("en-vi", "vi-en"):
            cfg = RunConfig().apply(json.loads((ROOT / f"configs/hy_a1_v13_{variant}.json").read_text()))
            cfg.apply({"backend": "mock", "run_id": f"preflight_{variant}_{direction}",
                       "glossary_path": str(ROOT / "configs/legal_glossary_v13.jsonl"),
                       "tokenizer_path": str(tok.path)})
            backend = MockBackend(tok=tok, batching=True)
            source = sources / direction.split("-")[0]
            rows = folder_rows(source, direction)
            assert len(rows) == 50
            out = build / f"preflight_{variant}_{direction}"
            summary = run_manifest(rows, source, out, cfg, backend, tok, directions=direction)
            assert not summary[direction]["statuses"].get("failed")
            assert len(list((out / "predictions" / direction).rglob("*.md"))) == 50
            terms = sum("translates to" in c["prompt"] for c in backend.calls)
            assert terms > 0 if variant == "glossary" else terms == 0
            rt = json.loads((out / "tests/parser_roundtrip.json").read_text())
            assert rt["all_roundtrip_ok"]
            report["runs"][f"{variant}/{direction}"] = {**summary[direction], "prompts_with_glossary": terms,
                                                         "all_roundtrip_ok": True}
            print(f"{variant} {direction}: 50 predictions, {terms} glossary prompts", flush=True)
    (ROOT / "reports/v13_mock_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

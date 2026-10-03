"""Refactor acceptance for code paths that have no recorded GPU trace (source context, soft glossary,
target history): run lam_docmt and pipeline-v1 docmt with their deterministic MockBackend (echo) on the
same inputs for every branch preset and compare predictions, document_status, traces and prompts.

  python tools/mock_equivalence.py --legacy-code ../report_lam_3_10/code \
      --manifest "../task 3-10/CHUNG/inputs/smoke6_doc_direction_manifest.jsonl" \
      --data-root "../task 3-10/CHUNG/inputs" --tokenizer ".../tokenizer.json" \
      --glossary "../task 3-10/CHUNG/inputs/release_v2/glossary_provenance.jsonl" \
      --work /tmp/mockeq --report reports/mock_equivalence.json
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from replay_equivalence import compare  # noqa: E402

BRANCHES = ("hy_a1_structure", "hy_a1_source_context", "hy_a1_source_context_glossary",
            "hy_a1_source_context_history")


def run(pkg, branch, args, rows, out):
    if pkg == "legacy":
        from lam_docmt.backends.mock import MockBackend
        from lam_docmt.config import make_config
        from lam_docmt.runtime import run_manifest
        from lam_docmt.tok import HyTok
    else:
        from docmt.backends.mock import MockBackend
        from docmt.core.config import make_config
        from docmt.engine.api import run_manifest
        from docmt.core.tok import HyTok
    cfg = make_config(branch, run_id=f"mockeq_{branch}", backend="mock", glossary_path=args.glossary,
                      extra={"resume": False})
    tok = HyTok(args.tokenizer)
    return run_manifest(rows, args.data_root, out, cfg, MockBackend(tok=tok), tok, "en-vi,vi-en",
                        manifest_path=args.manifest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("--legacy-code", "--manifest", "--data-root", "--tokenizer", "--glossary", "--work", "--report"):
        ap.add_argument(k, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.legacy_code).resolve()))
    rows = [json.loads(x) for x in Path(args.manifest).read_text(encoding="utf-8").splitlines() if x.strip()]
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    report = {"manifest": args.manifest, "branches": {}}
    for b in BRANCHES:
        run("legacy", b, args, rows, work / b / "legacy")
        run("v1", b, args, rows, work / b / "v1")
        report["branches"][b] = compare(work / b / "legacy", work / b / "v1")
    report["passed"] = all(v["passed"] for v in report["branches"].values())
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({b: {"passed": v["passed"], "predictions": v["predictions"]["identical"],
                          "trace_records": v["generation_trace"]["v1"], "trace_diff": v["generation_trace"]["different"]}
                      for b, v in report["branches"].items()}, indent=2))
    print("passed:", report["passed"])
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

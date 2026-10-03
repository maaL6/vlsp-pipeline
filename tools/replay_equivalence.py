"""Refactor acceptance without GPU: run the original lam_docmt and pipeline-v1 docmt on the same
inputs with the same config, both answered by the recorded outputs of a real run (ReplayBackend),
and compare their artifacts byte for byte.

  python tools/replay_equivalence.py \
      --legacy-code ../report_lam_3_10/code \
      --run-dir ../report_lam_3_10/delivery/hy_a1_structure__kaggleT4__20261003b \
      --manifest "../task 3-10/CHUNG/inputs/smoke6_doc_direction_manifest.jsonl" \
      --data-root "../task 3-10/CHUNG/inputs" \
      --tokenizer "../task 3-10/CHUNG/inputs/hy_a1_tokenizer/tokenizer.json" \
      --work /tmp/replay --report reports/replay_equivalence.json

Checks (all must hold for "passed"):
  1. predictions: identical bytes, every doc-direction (legacy vs v1)
  2. document_status.jsonl: identical records except wall-clock "seconds"
  3. generation_trace.jsonl: identical records (by request_id) except "seconds"
  4. prompts.jsonl: identical prompt set
Also reported (information, not a pass condition): how many legacy replay predictions equal the
predictions of the recorded run itself (the recorded run may predate later code fixes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent
sys.path.insert(0, str(V1))


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def legacy_replay_class(legacy_backends, legacy_types):
    from docmt.backends.replay import ReplayStore, replay_result

    class LegacyReplay(legacy_backends.Backend):
        name = "replay"
        batching = True

        def __init__(self, traces, cfg=None):
            super().__init__()
            self.cfg = cfg
            self.store = ReplayStore(traces)
            self.misses = []
            self.stats["replay_hit"] = 0
            self.stats["replay_miss"] = 0

        def decode_config(self):
            return {"backend": self.name, "do_sample": False,
                    "repetition_penalty": float(getattr(self.cfg, "repetition_penalty", 1.05))}

        def describe(self):
            return {"backend": self.name, "traces": self.store.sources, "recorded_prompts": len(self.store.by_sha),
                    "conflicts": self.store.conflicts, "parameter_count": None,
                    "parameter_count_source": "not_checked (replay, no model)", "decode": self.decode_config()}

        def generate(self, requests):
            out = []
            for r in requests:
                res, hit = replay_result(self.store, r, legacy_types.GenResult, self.name, self.misses)
                self.bump("replay_hit" if hit else "replay_miss")
                out.append(res)
            return out

    return LegacyReplay


def config_dict(run_dir: Path, run_id: str) -> dict:
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    cfg.pop("config_sha256", None)
    cfg.update(run_id=run_id, backend="mock", model="", model_sha256="", tokenizer_path="", tokenizer_sha256="")
    cfg["extra"] = dict(cfg.get("extra") or {}, resume=False)
    return cfg


def run_legacy(args, rows, cfg_d, traces, out):
    sys.path.insert(0, str(Path(args.legacy_code).resolve()))
    import lam_docmt.backends as LB
    import lam_docmt.types as LT
    from lam_docmt.config import RunConfig
    from lam_docmt.runtime import run_manifest
    from lam_docmt.tok import HyTok
    cfg = RunConfig().apply(cfg_d)
    backend = legacy_replay_class(LB, LT)(traces, cfg)
    summary = run_manifest(rows, args.data_root, out, cfg, backend, HyTok(args.tokenizer), "en-vi,vi-en",
                           manifest_path=args.manifest)
    return summary, backend


def run_v1(args, rows, cfg_d, traces, out):
    from docmt.backends.replay import ReplayBackend
    from docmt.core.config import RunConfig
    from docmt.core.tok import HyTok
    from docmt.engine.api import run_manifest
    cfg = RunConfig().apply(cfg_d)
    backend = ReplayBackend(traces, cfg)
    summary = run_manifest(rows, args.data_root, out, cfg, backend, HyTok(args.tokenizer), "en-vi,vi-en",
                           manifest_path=args.manifest)
    return summary, backend


def strip(rec, keys=("seconds",)):
    return {k: v for k, v in rec.items() if k not in keys}


def compare(a: Path, b: Path) -> dict:
    res = {}
    pa = sorted(p.relative_to(a) for p in (a / "predictions").rglob("*.md"))
    pb = sorted(p.relative_to(b) for p in (b / "predictions").rglob("*.md"))
    same = [str(p) for p in pa if p in pb and (a / p).read_bytes() == (b / p).read_bytes()]
    res["predictions"] = {"legacy": len(pa), "v1": len(pb), "identical": len(same),
                          "different": sorted(str(p) for p in set(pa) | set(pb) if str(p) not in same)}
    sa = {(r["direction"], r["doc_id"]): strip(r) for r in read_jsonl(a / "logs/document_status.jsonl")}
    sb = {(r["direction"], r["doc_id"]): strip(r) for r in read_jsonl(b / "logs/document_status.jsonl")}
    res["document_status"] = {"records": len(sa), "identical": sum(1 for k in sa if sb.get(k) == sa[k]),
                              "different": sorted("/".join(k) for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))}
    ta = {r["request_id"]: strip(r) for r in read_jsonl(a / "logs/generation_trace.jsonl")}
    tb = {r["request_id"]: strip(r) for r in read_jsonl(b / "logs/generation_trace.jsonl")}
    diff = sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))
    res["generation_trace"] = {"legacy": len(ta), "v1": len(tb), "different": len(diff), "examples": diff[:5]}
    qa = {r["prompt_sha256"] for r in read_jsonl(a / "logs/prompts.jsonl")}
    qb = {r["prompt_sha256"] for r in read_jsonl(b / "logs/prompts.jsonl")}
    res["prompts"] = {"legacy": len(qa), "v1": len(qb), "only_legacy": len(qa - qb), "only_v1": len(qb - qa)}
    res["passed"] = (not res["predictions"]["different"] and res["predictions"]["legacy"] > 0
                     and not res["document_status"]["different"] and not diff
                     and qa == qb)
    return res


def vs_recorded(out: Path, run_dir: Path) -> dict:
    rec = {(r["direction"], r["doc_id"]): r.get("output_sha256") for r in read_jsonl(run_dir / "logs/document_status.jsonl")}
    rows = []
    for (d, doc), sha in sorted(rec.items()):
        p = out / "predictions" / d / (doc + ".md")
        mine = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        rows.append({"direction": d, "doc_id": doc, "equal_to_recorded_run": mine == sha})
    return {"identical": sum(r["equal_to_recorded_run"] for r in rows), "documents": len(rows), "per_document": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--legacy-code", required=True)
    ap.add_argument("--run-dir", required=True, help="recorded run whose logs/generation_trace.jsonl is replayed")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    traces = [run_dir / "logs/generation_trace.jsonl"]
    rows = read_jsonl(args.manifest)
    work = Path(args.work)
    if work.exists():
        shutil.rmtree(work)
    cfg_d = config_dict(run_dir, "replay_equivalence")
    sum_a, be_a = run_legacy(args, rows, cfg_d, traces, work / "legacy")
    sum_b, be_b = run_v1(args, rows, cfg_d, traces, work / "v1")
    report = {
        "recorded_run": str(run_dir), "traces": [str(t) for t in traces],
        "legacy_code": str(args.legacy_code), "manifest": str(args.manifest),
        "config": cfg_d,
        "replay": {"legacy": {"hits": be_a.stats["replay_hit"], "misses": be_a.stats["replay_miss"],
                              "miss_request_ids": be_a.misses},
                   "v1": {"hits": be_b.stats["replay_hit"], "misses": be_b.stats["replay_miss"],
                          "miss_request_ids": be_b.misses}},
        "summary": {"legacy": sum_a, "v1": sum_b},
        "legacy_vs_v1": compare(work / "legacy", work / "v1"),
        "legacy_replay_vs_recorded_run": vs_recorded(work / "legacy", run_dir),
        "v1_replay_vs_recorded_run": vs_recorded(work / "v1", run_dir),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report["legacy_vs_v1"]["passed"], "legacy_vs_v1": report["legacy_vs_v1"],
                      "replay": {k: {"hits": v["hits"], "misses": v["misses"]} for k, v in report["replay"].items()},
                      "legacy_replay_vs_recorded_run": report["legacy_replay_vs_recorded_run"]["identical"],
                      "v1_replay_vs_recorded_run": report["v1_replay_vs_recorded_run"]["identical"]},
                     ensure_ascii=False, indent=2))
    return 0 if report["legacy_vs_v1"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

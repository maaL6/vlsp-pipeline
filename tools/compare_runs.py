# -*- coding: utf-8 -*-
"""Compare finished systems on smoke6 (post-generation only).

Per doc-direction and system: BLEU (13a, effective order), chrF++ and chrF against the RAW reference
(no normalisation), same sacrebleu metric classes as benchmark/eval_markdown.py. These are smoke
diagnostics on 6 risk-selected doc-directions, NOT headline scores and never used to tune anything.
Also: backend parity between two systems that share chunking/prompt (e.g. legacy on llama.cpp vs
legacy on Kaggle T4): chrF++ of one against the other, no gold involved.

  python tools/compare_runs.py --system B1=<root> --system a1_base=<root> ... --out tests_reports/compare_smoke6
  roots contain <direction>/<category>/<id>.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sacrebleu import __version__ as SB_VERSION
from sacrebleu.metrics import BLEU, CHRF

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
INPUTS = Path(os.environ.get("DOCMT_TEST_INPUTS") or CODE.parent / "task 3-10" / "CHUNG" / "inputs")


def read(p):
    return Path(p).read_text(encoding="utf-8") if Path(p).exists() else None


def scores(hyp, ref):
    b = BLEU(tokenize="13a", effective_order=True).sentence_score(hyp, [ref])
    c2 = CHRF(word_order=2).sentence_score(hyp, [ref])
    c0 = CHRF().sentence_score(hyp, [ref])
    return {"bleu": round(b.score, 2), "chrfpp": round(c2.score, 2), "chrf": round(c0.score, 2),
            "bp": round(b.bp, 3), "len_ratio": round(b.sys_len / max(b.ref_len, 1), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(INPUTS / "smoke6_doc_direction_manifest.jsonl"))
    ap.add_argument("--system", action="append", required=True, help="label=root")
    ap.add_argument("--parity", action="append", default=[], help="labelA:labelB (A scored against B)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    systems = dict(s.split("=", 1) for s in a.system)
    try:
        from docmt.steps.s10_doc_check import audit_plain_output
    except Exception:  # validator not available: metrics only
        audit_plain_output = None
    _b, _c = BLEU(tokenize="13a"), CHRF(word_order=2)
    _b.corpus_score(["a"], [["a"]]), _c.corpus_score(["a"], [["a"]])  # signatures need one evaluation
    report = {"sacrebleu": SB_VERSION, "bleu_signature": str(_b.get_signature()),
              "chrfpp_signature": str(_c.get_signature()),
              "note": "smoke6 diagnostic only (6 risk-selected doc-directions); reference read after generation; "
                      "not a headline score; not used for tuning", "docs": [], "corpus": {}, "parity": []}
    present = {}
    for r in rows:
        d = r["direction"]
        src = read(INPUTS / r["source_path"])
        ref = read(INPUTS / r["reference_path"])
        rel = Path(d, *r["sample_id"].split("/")).with_suffix(".md")
        entry = {"sample_id": r["sample_id"], "direction": d, "systems": {}}
        for label, root in systems.items():
            hyp = read(Path(root) / rel)
            if hyp is None:
                entry["systems"][label] = {"missing": True}
                continue
            present.setdefault((label, d), []).append(r["sample_id"])
            s = scores(hyp, ref)
            if audit_plain_output is not None:
                try:
                    au = audit_plain_output(src, hyp, d[:2], d[3:])
                    s["audit"] = {k: au[k] for k in au if k in (
                        "missing_heading_numbers", "extra_heading_numbers", "loops", "fffd",
                        "length_ratio", "big_number_diff", "internal_marker_leak")}
                except Exception as e:
                    s["audit_error"] = repr(e)
            entry["systems"][label] = s
        report["docs"].append(entry)
    # corpus scores per direction on the intersection of docs present for all systems
    for d in sorted({r["direction"] for r in rows}):
        common = None
        for label in systems:
            ids = set(present.get((label, d), []))
            common = ids if common is None else common & ids
        common = sorted(common or [])
        report["corpus"][d] = {"docs": common, "systems": {}}
        if not common:
            continue
        for label, root in systems.items():
            hyps, refs = [], []
            for r in rows:
                if r["direction"] == d and r["sample_id"] in common:
                    hyps.append(read(Path(root) / Path(d, *r["sample_id"].split("/")).with_suffix(".md")))
                    refs.append(read(INPUTS / r["reference_path"]))
            b = BLEU(tokenize="13a").corpus_score(hyps, [refs])
            c2 = CHRF(word_order=2).corpus_score(hyps, [refs])
            report["corpus"][d]["systems"][label] = {"bleu": round(b.score, 2), "chrfpp": round(c2.score, 2),
                                                     "bp": round(b.bp, 3)}
    for pair in a.parity:
        la, lb = pair.split(":")
        for r in rows:
            rel = Path(r["direction"], *r["sample_id"].split("/")).with_suffix(".md")
            ha, hb = read(Path(systems[la]) / rel), read(Path(systems[lb]) / rel)
            if ha is None or hb is None:
                continue
            s = scores(ha, hb)
            report["parity"].append({"pair": pair, "sample_id": r["sample_id"], "direction": r["direction"],
                                     "chrfpp_A_vs_B": s["chrfpp"], "bleu_A_vs_B": s["bleu"],
                                     "len_ratio_A_over_B": s["len_ratio"], "identical": ha == hb})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["| doc | dir | " + " | ".join(systems) + " |", "|---|---|" + "---|" * len(systems)]
    for e in report["docs"]:
        cells = []
        for label in systems:
            s = e["systems"][label]
            cells.append("missing" if s.get("missing") else f"{s['chrfpp']} / {s['bleu']}")
        lines.append(f"| {e['sample_id']} | {e['direction']} | " + " | ".join(cells) + " |")
    out.with_suffix(".md").write_text("chrF++ / BLEU per doc (smoke diagnostic)\n\n" + "\n".join(lines) + "\n",
                                      encoding="utf-8")
    print("\n".join(lines))
    print(json.dumps(report["corpus"], ensure_ascii=False, indent=1))
    for p in report["parity"]:
        print(p)


if __name__ == "__main__":
    main()

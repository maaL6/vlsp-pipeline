"""Chấm public test: SacreBLEU 2.6.0 corpus-level, mỗi tài liệu là một segment, text raw (không chuẩn hoá).

Thay cho benchmark/eval_markdown.py (không nằm trong repo git). Đã kiểm trên output B
(results/public_test_20261004/B_gom_node_khong_marker): BLEU 68.90 / chrF 83.39 en-vi và
46.62 / 79.17 vi-en, BP 0.942, ratio 0.943 — trùng số của eval_markdown.py.

python tools/score_public_test.py --csv dataset/public_test/en-vi-public_test.csv --hyp HYP_DIR [--save out.json]
HYP_DIR có HYP_DIR/vi/<sample_id>.md (chiều en-vi) và HYP_DIR/en/<sample_id>.md (chiều vi-en).
Thiếu file: chấm như bản dịch rỗng và liệt kê trong "missing".
"""
import argparse
import csv
import json
from pathlib import Path

import sacrebleu

csv.field_size_limit(10**9)
DIRS = {"en-vi": ("vi", "vietnamese_markdown"), "vi-en": ("en", "english_markdown")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--save")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))  # CRLF -> LF, như eval_markdown.py
    out = {"sacrebleu": sacrebleu.__version__, "unit": "document", "text": "raw"}
    for d, (sub, refcol) in DIRS.items():
        hyps, refs, missing, per_doc = [], [], [], []
        for r in rows:
            p = Path(a.hyp) / sub / f"{r['sample_id']}.md"
            h = p.read_text(encoding="utf-8") if p.exists() else ""
            if not p.exists():
                missing.append(r["sample_id"])
            hyps.append(h)
            refs.append(r[refcol])
            per_doc.append({"sample_id": r["sample_id"], "bleu": round(sacrebleu.sentence_bleu(h, [r[refcol]]).score, 2),
                            "len_ratio": round(len(h) / max(1, len(r[refcol])), 3)})
        metric = sacrebleu.BLEU()
        bleu = metric.corpus_score(hyps, [refs])
        out[d] = {"bleu": round(bleu.score, 2), "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
                  "chrf_pp": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
                  "bp": round(bleu.bp, 3), "ratio": round(bleu.sys_len / bleu.ref_len, 3),
                  "signature": str(metric.get_signature()), "docs": len(rows), "missing": missing,
                  "lowest5": sorted(per_doc, key=lambda x: x["bleu"])[:5]}
        print(f"{d}: BLEU {out[d]['bleu']:.2f}  chrF {out[d]['chrf']:.2f}  chrF++ {out[d]['chrf_pp']:.2f}  "
              f"BP {out[d]['bp']}  ratio {out[d]['ratio']}  missing {len(missing)}")
    if a.save:
        Path(a.save).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

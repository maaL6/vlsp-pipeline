"""Rescore saved translations and paired document bootstrap; never run inference.

Requires sacrebleu==2.6.0 and numpy. Paths are supplied explicitly so that output
from an earlier checkpoint cannot silently stand in for the submitted system.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import sacrebleu
from sacrebleu.metrics import BLEU, CHRF


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--abc-root", type=Path, required=True)
    ap.add_argument("--final-pred", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-resamples", type=int, default=1000)
    ap.add_argument("--final-resamples", type=int, default=2000)
    args = ap.parse_args()
    if sacrebleu.__version__ != "2.6.0":
        raise RuntimeError("Use sacrebleu==2.6.0 to reproduce these statistics")
    csv.field_size_limit(100_000_000)
    # Restore the official LF Markdown when a Windows CSV checkout uses CRLF.
    # This changes no lexical content and matches Path.read_text for hypotheses.
    with args.csv.open(encoding="utf-8-sig", newline=None) as f:
        rows = list(csv.DictReader(f))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != 50 or len(set(ids)) != 50:
        raise ValueError("Expected 50 unique paired document IDs")
    systems = {
        "A": args.abc_root / "A_gom_node_marker/predictions",
        "B": args.abc_root / "B_gom_node_khong_marker/predictions",
        "C": args.abc_root / "C_khong_gom_khong_marker/predictions",
        "final": args.final_pred,
    }
    report = {
        "policy": "raw Markdown with LF line endings; no lexical/Unicode normalisation; corpus metrics; document resampling",
        "sacrebleu_version": sacrebleu.__version__, "seed": args.seed,
        "csv": str(args.csv), "csv_sha256": digest(args.csv),
        "lf_csv_sha256": hashlib.sha256(args.csv.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        "sample_ids_in_resampling_order": ids,
        "group_resamples": args.group_resamples,
        "final_resamples": args.final_resamples,
        "percentile_method": "numpy.percentile, linear interpolation",
        "directions": {},
    }
    for direction, tgt, col in (
        ("en-vi", "vi", "vietnamese_markdown"),
        ("vi-en", "en", "english_markdown"),
    ):
        ref = [row[col] for row in rows]
        b, c = BLEU(tokenize="13a"), CHRF()
        scores, statistics = {}, {}
        for name, root in systems.items():
            paths = [root / tgt / (sid + ".md") for sid in ids]
            hypotheses = [p.read_text(encoding="utf-8") for p in paths]
            if any(not h.strip() for h in hypotheses):
                raise ValueError(f"Empty output: {name}/{direction}")
            bs, cs = b.corpus_score(hypotheses, [ref]), c.corpus_score(hypotheses, [ref])
            statistics[name] = np.asarray(b._extract_corpus_statistics(hypotheses, [ref]))
            scores[name] = {
                "bleu": bs.score, "chrf": cs.score,
                "prediction_root": str(root),
                "prediction_sha256_by_id": {sid: digest(p) for sid, p in zip(ids, paths)},
            }
            print(direction, name, round(bs.score, 4), round(cs.score, 4), flush=True)

        def from_stats(st):
            return b._compute_score_from_stats(st.sum(axis=0).tolist()).score

        rng = np.random.default_rng(args.seed)
        paired = {}
        draws = rng.integers(0, len(ids), size=(args.group_resamples, len(ids)))
        for high, low in (("B", "A"), ("A", "C"), ("B", "C")):
            delta = np.array([
                from_stats(statistics[high][idx]) - from_stats(statistics[low][idx])
                for idx in draws
            ])
            paired[f"{high}-{low}"] = {
                "delta_bleu": scores[high]["bleu"] - scores[low]["bleu"],
                "ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
                "ci_excludes_zero": bool(
                    np.percentile(delta, 2.5) > 0 or np.percentile(delta, 97.5) < 0
                ),
            }
        rng = np.random.default_rng(args.seed)
        final_draws = rng.integers(0, len(ids), size=(args.final_resamples, len(ids)))
        final_scores = [from_stats(statistics["final"][idx]) for idx in final_draws]
        report["directions"][direction] = {
            "systems": scores, "paired_grouping": paired,
            "final_bleu_ci95": np.percentile(final_scores, [2.5, 97.5]).tolist(),
            "signatures": {"bleu": str(b.get_signature()), "chrf": str(c.get_signature())},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

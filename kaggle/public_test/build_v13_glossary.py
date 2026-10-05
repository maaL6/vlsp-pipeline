"""Retain reviewed glossary pairs with bilingual witnesses in train-only rows."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from docmt.context.common import _fold
from docmt.core.text_normalize import normalize


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glossary", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev-ids", required=True)
    ap.add_argument("--public-csv", required=True)
    ap.add_argument("--out", default=str(ROOT / "configs/legal_glossary_v13.jsonl"))
    args = ap.parse_args()
    dev = set(json.loads(Path(args.dev_ids).read_text(encoding="utf-8")))
    csv.field_size_limit(10_000_000)
    with Path(args.public_csv).open(encoding="utf-8-sig", newline="") as handle:
        public = {row["sample_id"] for row in csv.DictReader(handle)}
    entries, seen = [], set()
    for line in Path(args.glossary).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("status") != "verified":
            continue
        row = dict(row, en=normalize(row["en"]).strip(), vi=normalize(row["vi"]).strip())
        key = (_fold(row["en"]), _fold(row["vi"]))
        if key not in seen and min(map(len, key)) >= 3:
            entries.append(row)
            seen.add(key)
    tables = {lang: collections.defaultdict(set) for lang in ("en", "vi")}
    for i, entry in enumerate(entries):
        for lang in tables:
            tables[lang][_fold(entry[lang])].add(i)
    patterns = {lang: re.compile(r"(?<!\w)(?:" + "|".join(
        re.escape(term) for term in sorted(table, key=len, reverse=True)) + r")(?!\w)")
        for lang, table in tables.items()}
    witnesses, matched_docs = {}, collections.defaultdict(set)
    train_docs, rows = set(), 0
    with Path(args.train).open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            row = json.loads(line)
            sid = row["sample_id"]
            if sid in dev or sid in public:
                raise ValueError(f"evaluation document in train rows: {sid}")
            train_docs.add(sid)
            rows += 1
            matches = {}
            for lang, field in (("en", "english"), ("vi", "vietnamese")):
                found = set()
                text = _fold(normalize(row[field]))
                for match in patterns[lang].finditer(text):
                    found.update(tables[lang][match.group()])
                matches[lang] = found
            for i in matches["en"] & matches["vi"]:
                witnesses.setdefault(i, {"sample_id": sid, "train_line": lineno,
                                         "view": row.get("view"), "section_id": row.get("section_id")})
                matched_docs[i].add(sid)
    kept = []
    for i, entry in enumerate(entries):
        if i in witnesses:
            kept.append(dict(entry, audit_status="reviewed_pair_with_train_witness",
                             historical_origin="reviewed_local_glossary; bilingual train-only co-occurrence",
                             train_witness=witnesses[i], train_document_count=len(matched_docs[i])))
    if not kept:
        raise ValueError("no glossary pairs have train-only witnesses")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")
    audit = {"reviewed_candidates": len(entries), "kept": len(kept), "dropped_no_train_witness": len(entries)-len(kept),
             "train_rows": rows, "train_documents": len(train_docs), "dev_documents": len(dev),
             "public_documents": len(public), "train_dev_overlap": 0, "train_public_overlap": 0,
             "source_glossary_sha256": sha(args.glossary), "train_sha256": sha(args.train),
             "dev_ids_sha256": sha(args.dev_ids), "public_csv_sha256": sha(args.public_csv),
             "output_sha256": sha(out), "reference_targets_read": False,
             "caveat": "Witnesses establish train provenance/co-occurrence, not a new semantic expert review."}
    (ROOT / "reports/glossary_v13_audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()

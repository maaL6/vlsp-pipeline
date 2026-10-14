"""Write the source documents of a private test CSV as .md files, one folder per source language.

  python tools/export_sources.py --csv dataset/private_test/private_test_input.csv --out work/sources

Creates OUT/en/<sample_id>.md and OUT/vi/<sample_id>.md (the input folders of `run.sh`).
"""
import argparse
import csv
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=Path("dataset/private_test/private_test_input.csv"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    csv.field_size_limit(100_000_000)
    counts = {"en": 0, "vi": 0}
    with args.csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            lang = row["source_language"].strip().lower()
            sid = Path(row["sample_id"])
            if lang not in counts or sid.is_absolute() or ".." in sid.parts:
                raise SystemExit(f"bad row: {row['sample_id']!r} ({lang!r})")
            path = args.out / lang / (row["sample_id"] + ".md")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(row["source_markdown"].encode("utf-8"))
            counts[lang] += 1
    print(f"wrote {counts['en']} en and {counts['vi']} vi documents to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

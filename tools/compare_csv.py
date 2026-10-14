"""Compare a result CSV (sample_id, direction, translation) with an expected one: identical rows and chrF.

  python tools/compare_csv.py results.csv examples/demo_expected_kaggle.csv [--min-chrf 90]

Exit code 1 if ids/directions differ, a translation is empty, or chrF is below --min-chrf. chrF needs sacrebleu.
"""
import argparse
import csv
import sys
from pathlib import Path


def read(path: Path) -> dict:
    csv.field_size_limit(100_000_000)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {(r["sample_id"], r["direction"]): r["translation"] for r in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result", type=Path)
    ap.add_argument("expected", type=Path)
    ap.add_argument("--min-chrf", type=float, default=0.0)
    args = ap.parse_args()
    got, want = read(args.result), read(args.expected)
    ok = set(got) == set(want)
    if not ok:
        print(f"keys differ: missing {sorted(set(want) - set(got))[:5]}, extra {sorted(set(got) - set(want))[:5]}")
    keys = [k for k in want if k in got]
    empty = [k for k in keys if not got[k].strip()]
    same = sum(got[k] == want[k] for k in keys)
    print(f"{len(keys)} rows compared: {same} identical, {len(empty)} empty")
    ok = ok and not empty
    try:
        from sacrebleu.metrics import CHRF
        chrf = CHRF().corpus_score([got[k] for k in keys], [[want[k] for k in keys]]).score
        print(f"chrF vs expected: {chrf:.2f}")
        ok = ok and chrf >= args.min_chrf
    except ImportError:
        print("sacrebleu not installed: chrF skipped (pip install sacrebleu)")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

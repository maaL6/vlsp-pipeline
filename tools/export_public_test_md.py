"""Public test CSV (BTC) -> OUT/en/<sample_id>.md, OUT/vi/<sample_id>.md, byte-exact Markdown.

Thay cho benchmark/export_public_test.py (benchmark/ không nằm trong repo git)."""
import argparse
import csv
from pathlib import Path

csv.field_size_limit(10**9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n = 0
    # CSV dùng CRLF; đọc mặc định (CRLF -> LF) để file nguồn khớp sha nguồn run B của Lâm (50/50 mỗi chiều)
    for r in csv.DictReader(open(a.csv, encoding="utf-8")):
        for lang, col in (("en", "english_markdown"), ("vi", "vietnamese_markdown")):
            p = Path(a.out) / lang / f"{r['sample_id']}.md"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(r[col])
        n += 1
    print(f"{n} docs -> {a.out}/en, {a.out}/vi")


if __name__ == "__main__":
    main()

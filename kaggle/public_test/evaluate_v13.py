"""Re-score predictions using the unchanged downloaded evaluator (Windows CSV limit shim)."""
import csv
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = Path(sys.argv[1]).resolve()
    bench = ROOT / "kaggle/public_test/build/original/benchmark"
    field_limit = csv.field_size_limit
    if sys.platform == "win32":
        # CPython on Windows has a 32-bit C long; sys.maxsize cannot be passed directly.
        def compatible_limit(limit=None):
            return field_limit() if limit is None else field_limit(min(limit, 2**31 - 1))
        csv.field_size_limit = compatible_limit
    sys.path.insert(0, str(bench))
    status = __import__("json").loads((out / "kernel_status.json").read_text(encoding="utf-8"))
    for variant in ("batch_length", "glossary"):
        if not status.get("benchmark", {}).get(variant, {}).get("complete_50x2"):
            print(f"Skip incomplete benchmark: {variant}")
            continue
        sys.argv = [str(bench / "eval_markdown.py"), "--csv", str(bench / "data/en-vi-public_test.csv"),
                    "--direction", "both", "--hyp", str(out / f"hyp_{variant}"),
                    "--save", str(out / f"scores_local_{variant}.json")]
        runpy.run_path(str(bench / "eval_markdown.py"), run_name="__main__")


if __name__ == "__main__":
    main()

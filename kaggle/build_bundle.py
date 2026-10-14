"""Stage the runtime dataset that kernel_private.py reads from /kaggle/input.

  python kaggle/build_bundle.py --csv PRIVATE_INPUT.csv --out kaggle/build/runtime [--owner OWNER --slug SLUG]

Layout written to --out (must not exist yet):
  pipeline-v1/{docmt,configs,run.sh,requirements.txt}   code of this repo
  calibration/{en-vi,vi-en}/config.json                  per-direction run config (batch 8)
  data/private_test_input.csv                            the private test input (sample_id, category, source_language, source_markdown)
  run_spec.json                                          kaggle/run_spec.json with git_head and csv_md5 filled in
  pipeline_bundle_manifest.json                          sha256 of every file above; the kernel verifies it
  dataset-metadata.json                                  only with --owner/--slug (for `kaggle datasets create`)
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KAGGLE = REPO / "kaggle"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True, help="private test input CSV")
    ap.add_argument("--out", type=Path, required=True, help="directory to create")
    ap.add_argument("--owner", help="Kaggle username (writes dataset-metadata.json)")
    ap.add_argument("--slug", help="Kaggle dataset slug (writes dataset-metadata.json)")
    args = ap.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} already exists; remove it or choose another --out")
    if not args.csv.is_file():
        sys.exit(f"CSV not found: {args.csv}")

    out = args.out
    pipe = out / "pipeline-v1"
    pipe.mkdir(parents=True)
    copy_tree(REPO / "docmt", pipe / "docmt")
    copy_tree(REPO / "configs", pipe / "configs")
    for name in ("run.sh", "requirements.txt"):
        shutil.copyfile(REPO / name, pipe / name)
    copy_tree(KAGGLE / "calibration", out / "calibration")
    (out / "data").mkdir()
    shutil.copyfile(args.csv, out / "data" / "private_test_input.csv")

    git_head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "docmt", "configs", "run.sh",
                                 "requirements.txt", "kaggle"], capture_output=True, text=True).stdout.strip())
    spec = json.loads((KAGGLE / "run_spec.json").read_text(encoding="utf-8"))
    spec["git_head"] = git_head
    spec["csv_md5"] = hashlib.md5((out / "data" / "private_test_input.csv").read_bytes()).hexdigest()
    (out / "run_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    files = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    manifest = {"git_head": git_head, "uncommitted_checkout_used": dirty, "files": files}
    (out / "pipeline_bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.owner and args.slug:
        meta = {"title": args.slug, "id": f"{args.owner}/{args.slug}", "licenses": [{"name": "CC0-1.0"}], "isPrivate": True}
        (out / "dataset-metadata.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    print(f"bundle: {out} ({len(files)} files, git_head {git_head[:8]}{' + uncommitted changes' if dirty else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

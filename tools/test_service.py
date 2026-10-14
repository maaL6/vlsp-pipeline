"""Smoke / regression test of a running translation service (the Docker image).

  python tools/test_service.py --addr localhost:8000 [--mode public|private] [--n 3] [--ids ID,ID] [--out work/service_test]

public  (default) translates each selected public-test document in both directions and scores the output
        against the reference side of dataset/public_test/en-vi-public_test.csv (BLEU / chrF, needs sacrebleu).
private translates the selected private-test documents and compares the output with the Kaggle result in
        submission.zip (exact match rate and chrF between the two).

Every request must return HTTP 200 with the expected detected direction and a non-empty UTF-8 body.
By default the n shortest documents are used (fast); pass --n 0 for all of them.
Exit code 0 only if every request succeeded (and, in private mode, --min-chrf is reached).
"""
import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never use an HTTP proxy for the service


def wait_ready(addr: str, seconds: int) -> None:
    deadline = time.time() + seconds
    while True:
        try:
            with OPENER.open(f"http://{addr}/health", timeout=10) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        if time.time() > deadline:
            sys.exit(f"service at {addr} is not ready")
        time.sleep(2)


def translate(addr: str, text: str) -> tuple[int, dict, str]:
    request = urllib.request.Request(f"http://{addr}/translate", data=text.encode("utf-8"), method="POST",
                                     headers={"Content-Type": "text/markdown; charset=utf-8"})
    try:
        with OPENER.open(request) as resp:
            return resp.status, dict(resp.headers), resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read().decode("utf-8", "replace")


def read_csv(path: Path) -> list:
    csv.field_size_limit(100_000_000)
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pick(items: list, n: int, ids: str | None, key) -> list:
    if ids:
        wanted = [x.strip() for x in ids.split(",") if x.strip()]
        return [x for x in items if key(x) in wanted]
    items = sorted(items, key=lambda x: len(x["_text"]))
    return items if n == 0 else items[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--addr", default="localhost:8000")
    ap.add_argument("--mode", choices=("public", "private"), default="public")
    ap.add_argument("--csv", type=Path, help="default: dataset/public_test/en-vi-public_test.csv or dataset/private_test/private_test_input.csv")
    ap.add_argument("--baseline", type=Path, default=ROOT / "submission.zip", help="private mode: Kaggle results to compare with")
    ap.add_argument("--n", type=int, default=3, help="documents per language (0 = all)")
    ap.add_argument("--ids", help="comma separated sample_ids instead of the n shortest")
    ap.add_argument("--out", type=Path, default=Path("work/service_test"))
    ap.add_argument("--wait", type=int, default=1800, help="seconds to wait for the service")
    ap.add_argument("--min-chrf", type=float, default=0.0, help="private mode: fail below this chrF vs the baseline")
    args = ap.parse_args()

    wait_ready(args.addr, args.wait)
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError:
        BLEU = CHRF = None
        print("sacrebleu not installed: quality scores skipped (pip install sacrebleu)")

    jobs = []  # (sample_id, expected direction, source text, reference text)
    if args.mode == "public":
        rows = read_csv(args.csv or ROOT / "dataset" / "public_test" / "en-vi-public_test.csv")
        for r in rows:
            r["_text"] = r["english_markdown"]
        for r in pick(rows, args.n, args.ids, lambda r: r["sample_id"]):
            jobs.append((r["sample_id"], "en-vi", r["english_markdown"], r["vietnamese_markdown"]))
            jobs.append((r["sample_id"], "vi-en", r["vietnamese_markdown"], r["english_markdown"]))
    else:
        rows = read_csv(args.csv or ROOT / "dataset" / "private_test" / "private_test_input.csv")
        with zipfile.ZipFile(args.baseline) as z:
            baseline = {(r["sample_id"], r["direction"]): r["translation"]
                        for r in csv.DictReader(io.StringIO(z.read("results.csv").decode("utf-8")))}
        for r in rows:
            r["_text"] = r["source_markdown"]
        for lang, direction in (("en", "en-vi"), ("vi", "vi-en")):
            subset = [r for r in rows if r["source_language"].strip().lower() == lang]
            for r in pick(subset, args.n, args.ids, lambda r: r["sample_id"]):
                jobs.append((r["sample_id"], direction, r["source_markdown"], baseline[(r["sample_id"], direction)]))

    failures, hyps, refs, exact = 0, {"en-vi": [], "vi-en": []}, {"en-vi": [], "vi-en": []}, 0
    print(f"{'document':32} {'dir':6} {'status':>8} {'words':>7} {'seconds':>8} {'w/s':>7}  result")
    for sid, direction, source, reference in jobs:
        started = time.time()
        code, headers, body = translate(args.addr, source)
        elapsed = time.time() - started
        h = {k.lower(): v for k, v in headers.items()}
        problems = []
        if code != 200:
            problems.append(f"HTTP {code}: {body[:200]}")
        else:
            if h.get("x-direction") != direction:
                problems.append(f"detected {h.get('x-direction')} instead of {direction}")
            if not body.strip():
                problems.append("empty translation")
        note = "ok"
        if not problems:
            out = args.out / direction / (sid + ".md")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(body.encode("utf-8"))
            hyps[direction].append(body)
            refs[direction].append(reference)
            if args.mode == "private":
                same = body == reference
                exact += same
                note = "identical to Kaggle" if same else "differs from Kaggle"
            if h.get("x-status") not in (None, "completed"):
                note += f" ({h['x-status']})"
        else:
            failures += 1
            note = "FAILED: " + "; ".join(problems)
        print(f"{sid:32} {direction:6} {code:>8} {len(source.split()):>7} {elapsed:>8.1f} "
              f"{len(source.split()) / max(elapsed, 1e-9):>7.1f}  {note}", flush=True)

    ok = failures == 0
    if BLEU is not None:
        for direction in ("en-vi", "vi-en"):
            if not hyps[direction]:
                continue
            bleu = BLEU(tokenize="13a").corpus_score(hyps[direction], [refs[direction]]).score
            chrf = CHRF().corpus_score(hyps[direction], [refs[direction]]).score
            label = "vs reference" if args.mode == "public" else "vs Kaggle results"
            print(f"{direction}: {len(hyps[direction])} documents, BLEU {bleu:.2f}, chrF {chrf:.2f} ({label})")
            if args.mode == "private" and chrf < args.min_chrf:
                ok = False
                print(f"  chrF below --min-chrf {args.min_chrf}")
    if args.mode == "private":
        print(f"identical to the Kaggle output: {exact}/{len(jobs) - failures}")
    print(("PASS" if ok else "FAIL") + f" ({len(jobs) - failures}/{len(jobs)} requests succeeded; outputs in {args.out})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

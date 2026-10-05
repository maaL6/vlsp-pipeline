"""Two ablations on one T4 x2 machine; each GPU runs EN->VI then VI->EN."""
import datetime
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path

MODEL_SHA = "5dc305ae6c33c15a8440a4a8f9183e411004079eb737bc2da417b6a932dead81"
CSV_SHA = "dd7e0070a735da6a82de1c5c426afb8fdc4a9b0bf16c1271cf2396ddb95d4ef3"
CONFIGS = {"batch_length": "configs/hy_a1_v13_batch_length.json",
           "glossary": "configs/hy_a1_v13_glossary.json"}
GPU_MAP = {"batch_length": 0, "glossary": 1}
DIRECTIONS = ("en-vi", "vi-en")
W = Path("/kaggle/working")
DEADLINE_S = 11 * 3600 + 15 * 60


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def find_one(pattern):
    hits = sorted(set(glob.glob(pattern, recursive=True)))
    assert len(hits) == 1, (pattern, hits)
    return Path(hits[0])


def command(variant, direction, pipe, model, sources):
    src, tgt = direction.split("-")
    return [sys.executable, "-m", "docmt", "run-dir",
            "--input-dir", str(sources / src), "--output-dir", str(W / f"hyp_{variant}" / tgt),
            "--work-dir", str(W / f"work_{variant}_{direction}"), "--direction", direction,
            "--config", str(pipe / CONFIGS[variant]), "--run-id", f"v13_{variant}_pt50_20261004",
            "--backend", "hf", "--model", str(model), "--tokenizer", str(model / "tokenizer.json"),
            "--set", "batch_size=4"]


def main():
    t0 = time.monotonic()
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.8.1",
                    "tokenizers>=0.21", "accelerate", "safetensors", "sentencepiece", "sacrebleu>=2.3"], check=True)
    root = find_one("/kaggle/input/**/pipeline_bundle_manifest.json").parent
    manifest = json.loads((root / "pipeline_bundle_manifest.json").read_text())
    assert all(sha(root / rel) == digest for rel, digest in manifest["files"].items()), "bundle hash mismatch"
    code = W / "code"
    shutil.copytree(root, code)
    pipe, bench = code / "pipeline-v1", code / "benchmark"
    weights = find_one("/kaggle/input/**/model.safetensors")
    assert sha(weights) == MODEL_SHA, "model hash mismatch"
    model = weights.parent
    import torch
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    assert len(names) == 2 and all("T4" in n for n in names), f"requires T4 x2, got {names}"
    csv = bench / "data/en-vi-public_test.csv"
    assert sha(csv) == CSV_SHA, "CSV hash mismatch"
    sources = W / "public_test_md"
    subprocess.run([sys.executable, str(bench / "export_public_test.py"), "--csv", str(csv),
                    "--out", str(sources)], check=True)
    assert all(len(list((sources / lang).rglob("*.md"))) == 50 for lang in ("en", "vi"))
    status = {"experiment": "v1.3_two_ablations", "batch_size": 4, "gpu_names": names,
              "gpu_map": GPU_MAP, "directions_order": list(DIRECTIONS), "model_sha256": MODEL_SHA,
              "csv_sha256": CSV_SHA, "bundle_manifest_sha256": sha(root / "pipeline_bundle_manifest.json"),
              "git_head": manifest["git_head"], "reference_used_by_pipeline": False,
              "utc_started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "runs": {}, "benchmark": {}}
    def save():
        tmp = W / "kernel_status.tmp"
        tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        tmp.replace(W / "kernel_status.json")
    active = {}
    queue = {v: list(DIRECTIONS) for v in GPU_MAP}
    def start(v):
        d = queue[v].pop(0)
        env = dict(os.environ, PYTHONPATH=str(pipe), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   CUDA_VISIBLE_DEVICES=str(GPU_MAP[v]), TOKENIZERS_PARALLELISM="false")
        log = open(W / f"run_{v}_{d}.log", "w")
        p = subprocess.Popen(command(v, d, pipe, model, sources), cwd=pipe, env=env,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        active[v] = (d, p, log, time.monotonic())
        status["runs"][f"{v}/{d}"] = {"gpu": GPU_MAP[v], "state": "running", "config": CONFIGS[v]}
        save()
        print(f"START {v} {d} GPU={GPU_MAP[v]}", flush=True)
    def finish(v, timeout=False):
        d, p, log, began = active.pop(v)
        log.close()
        work = W / f"work_{v}_{d}"
        pred = work / "predictions" / d
        hyp = W / f"hyp_{v}" / d.split("-")[1]
        files = list(pred.rglob("*.md")) if pred.exists() else []
        for f in files:
            dst = hyp / f.relative_to(pred)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
        runfile = work / "logs" / f"{d}.run.json"
        run = json.loads(runfile.read_text()) if runfile.exists() else {}
        st = status["runs"][f"{v}/{d}"]
        complete = not timeout and p.returncode == 0 and len(files) == 50 and run.get("documents") == 50
        st.update(state="completed" if complete else ("timeout" if timeout else "failed"),
                  returncode=p.returncode, wall_seconds=round(time.monotonic() - began, 3),
                  predictions=len(files), expected=50, run=run)
        save()
        print(f"END {v} {d}: {st['state']}, predictions={len(files)}/50", flush=True)
    for v in GPU_MAP:
        start(v)
    while active:
        time.sleep(15)
        expired = time.monotonic() - t0 >= DEADLINE_S
        for v, (d, p, _, _) in list(active.items()):
            if expired and p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(45)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
            if p.poll() is not None:
                finish(v, expired)
                if queue[v] and not expired:
                    start(v)
        if expired:
            break
    for v in GPU_MAP:
        for d in queue[v]:
            status["runs"][f"{v}/{d}"] = {"state": "not_started", "reason": "deadline"}
        complete = all(status["runs"].get(f"{v}/{d}", {}).get("state") == "completed" for d in DIRECTIONS)
        score = W / f"scores_{v}.json"
        args = [sys.executable, str(bench / "eval_markdown.py"), "--csv", str(csv),
                "--direction", "both", "--hyp", str(W / f"hyp_{v}"), "--save", str(score)]
        if not complete:
            args.append("--allow-missing")
        with open(W / f"scores_{v}.txt", "w") as out:
            rc = subprocess.run(args, cwd=bench, stdout=out, stderr=subprocess.STDOUT).returncode
        status["benchmark"][v] = {"complete_50x2": complete, "returncode": rc, "scores": score.name}
        print((W / f"scores_{v}.txt").read_text()[-2500:], flush=True)
    status["utc_finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    with zipfile.ZipFile(W / "results_v13.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in W.iterdir():
            if path.name.startswith(("hyp_", "work_")):
                for f in path.rglob("*"):
                    if f.is_file():
                        z.write(f, f.relative_to(W).as_posix())
            elif path.suffix in (".json", ".txt", ".log"):
                z.write(path, path.name)
    shutil.rmtree(code)
    shutil.rmtree(sources)
    print(json.dumps(status, indent=2), flush=True)
    if not all(x["complete_50x2"] and x["returncode"] == 0 for x in status["benchmark"].values()):
        raise RuntimeError("benchmark incomplete; partial outputs preserved")


if __name__ == "__main__":
    main()

"""Kaggle kernel: full pipeline-v1 (HY A1, HF backend) on the VLSP26 public test, both directions,
then SacreBLEU with benchmark/eval_markdown.py.

VARIANT (set by push.sh when the kernel file is generated):
  markers    (A) configs/hy_a1_structure.json: units of up to 8 nodes (16 table cells), each node
             prefixed with [SEG_000123] (a unit with a single node is sent bare)
  nomarker   (C) configs/hy_a1_structure_nomarker.json: max_nodes=1, max_table_nodes=1, i.e. every node
             (or split piece) is its own request and no marker is ever inserted
  newline    (B, chosen) configs/hy_a1_structure_newline.json: units packed exactly like "markers", but the
             nodes are joined by newlines only (no marker); the output is split back by line and a
             line-count mismatch sends the unit to recovery (each node alone)

T4 x2: en-vi on GPU 0 and vi-en on GPU 1 in parallel (one GPU: sequential). A deadline stops the
runs before the 12 h session limit; the request cache in work_<direction>/ is kept in the output, and
a later version that mounts this kernel's output (or any input holding work_<variant>_<dir>/)
resumes from it.
"""
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

VARIANT = "__VARIANT__"
RUN_IDS = {"markers": "hy_a1_structure__pt50__kaggleT4__20261004",
           "nomarker": "hy_a1_structure_nomarker__pt50__kaggleT4__20261004",
           "newline": "hy_a1_structure_newline__pt50__kaggleT4__20261004"}
CONFIGS = {"markers": "configs/hy_a1_structure.json",
           "nomarker": "configs/hy_a1_structure_nomarker.json",
           "newline": "configs/hy_a1_structure_newline.json"}
MODEL_SHA256 = "5dc305ae6c33c15a8440a4a8f9183e411004079eb737bc2da417b6a932dead81"
BATCH_SIZE = int(os.environ.get("DOCMT_BATCH_SIZE", "4"))    # same as the verified smoke runs
DEADLINE_S = 11 * 3600 + 15 * 60                               # Kaggle GPU session limit is 12 h
DIRECTIONS = ("en-vi", "vi-en")
W = Path("/kaggle/working")
T0 = time.time()


def log(msg):
    print(f"[{datetime.datetime.now(datetime.timezone.utc):%H:%M:%S}Z +{time.time() - T0:7.0f}s] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def find_one(pattern):
    hits = sorted(set(glob.glob(pattern, recursive=True)))
    assert len(hits) == 1, (pattern, hits)
    return Path(hits[0])


def main():
    assert VARIANT in RUN_IDS, VARIANT
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.8.1", "tokenizers>=0.21",
                    "accelerate", "safetensors", "sentencepiece", "sacrebleu>=2.3"], check=True)

    # -- code bundle (verified against its manifest) ------------------------------------------
    root = find_one("/kaggle/input/**/pipeline_bundle_manifest.json").parent
    manifest = json.loads((root / "pipeline_bundle_manifest.json").read_text())
    bad = [rel for rel, sha in manifest["files"].items() if sha256_file(root / rel) != sha]
    assert not bad, f"bundle files differ from manifest: {bad[:5]}"
    code = W / "code"                       # writable copy (python caches, eval outputs)
    if code.exists():
        shutil.rmtree(code)
    shutil.copytree(root, code)
    pipe, bench = code / "pipeline-v1", code / "benchmark"

    # -- model ----------------------------------------------------------------------------------
    weights = find_one("/kaggle/input/**/model.safetensors")
    model = weights.parent
    model_sha = sha256_file(weights)
    assert model_sha == MODEL_SHA256, f"unexpected A1 weights {model_sha}"
    import torch
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "no CUDA GPU (enable GPU T4 x2 in the kernel settings)"
    log(f"variant={VARIANT} gpus={n_gpu} model={model} bundle={manifest.get('git_head')}")

    # -- public test -> .md sources (benchmark/export_public_test.py, original BTC CSV) -----------
    csv_path = bench / "data" / "en-vi-public_test.csv"
    assert hashlib.md5(csv_path.read_bytes()).hexdigest() == "d86e055d3f84436ae07e6c27fe686bcc"
    src_md = W / "public_test_md"
    subprocess.run([sys.executable, str(bench / "export_public_test.py"), "--csv", str(csv_path),
                    "--out", str(src_md)], check=True)

    hyp = W / f"hyp_{VARIANT}"              # eval_markdown layout: hyp/vi/... (en-vi), hyp/en/... (vi-en)
    status = dict(variant=VARIANT, run_id=RUN_IDS[VARIANT], config=CONFIGS[VARIANT], batch_size=BATCH_SIZE,
                  model_sha256=model_sha, bundle=manifest.get("git_head"), gpu_count=n_gpu,
                  gpu_name=torch.cuda.get_device_name(0), reference_used_by_pipeline=False,
                  utc_started=datetime.datetime.now(datetime.timezone.utc).isoformat(), directions={})

    def save():
        (W / "kernel_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False))

    # -- resume: copy any earlier work dir of the same variant/direction ------------------------
    for d in DIRECTIONS:
        work = W / f"work_{VARIANT}_{d}"
        prev = sorted(p for p in glob.glob(f"/kaggle/input/**/work_{VARIANT}_{d}/logs/cache.jsonl", recursive=True))
        if prev and not work.exists():
            shutil.copytree(Path(prev[-1]).parent.parent, work)
            log(f"resume {d}: copied {Path(prev[-1]).parent.parent}")

    # -- run both directions ------------------------------------------------------------------
    procs = {}

    def start(d, gpu):
        src, tgt = d.split("-")
        cmd = [sys.executable, "-m", "docmt", "run-dir", "--input-dir", str(src_md / src),
               "--output-dir", str(hyp / tgt), "--work-dir", str(W / f"work_{VARIANT}_{d}"),
               "--direction", d, "--config", str(pipe / CONFIGS[VARIANT]), "--run-id", RUN_IDS[VARIANT],
               "--backend", "hf", "--model", str(model), "--tokenizer", str(model / "tokenizer.json"),
               "--set", f"batch_size={BATCH_SIZE}"]
        env = dict(os.environ, PYTHONPATH=str(pipe), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
        logf = open(W / f"run_{VARIANT}_{d}.log", "w")
        p = subprocess.Popen(cmd, cwd=pipe, env=env, stdout=logf, stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs[d] = (p, logf, time.time())
        status["directions"][d] = dict(gpu=gpu, state="running")
        save()
        log(f"start {d} on GPU {gpu}")

    def finish(d, timed_out):
        p, logf, t = procs.pop(d)
        logf.close()
        work = W / f"work_{VARIANT}_{d}"
        tgt = d.split("-")[1]
        # predictions written so far (run-dir only copies them at the end; do it here too)
        n_pred = 0
        for f in (work / "predictions" / d).rglob("*.md") if (work / "predictions" / d).exists() else []:
            dst = hyp / tgt / f.relative_to(work / "predictions" / d)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
            n_pred += 1
        docs = [json.loads(x) for x in (work / "logs/document_status.jsonl").read_text().splitlines()] \
            if (work / "logs/document_status.jsonl").exists() else []
        run_json = work / "logs" / f"{d}.run.json"
        rj = json.loads(run_json.read_text()) if run_json.exists() else {}
        keep = ("documents", "nodes", "units", "requests", "requests_sent", "cache_hits", "recovery_requests",
                "retry_count", "runtime", "prompt_tokens", "generated_tokens", "peak_vram", "oom_count",
                "op_error_count", "unresolved_units", "unresolved_empty_nodes")
        st = {}
        for r in docs:
            st[r.get("status")] = st.get(r.get("status"), 0) + 1
        status["directions"][d].update(
            state="timeout" if timed_out else "finished", returncode=p.returncode,
            seconds=round(time.time() - t, 1), predictions=n_pred, expected=50,
            document_status=st, run=dict((k, rj.get(k)) for k in keep))
        save()
        log(f"end {d}: {status['directions'][d]}")

    queue = list(DIRECTIONS)
    gpus = list(range(min(n_gpu, 2)))
    for g in gpus:
        if queue:
            start(queue.pop(0), g)
    while procs:
        time.sleep(30)
        for d, (p, _, _) in list(procs.items()):
            if p.poll() is not None:
                gpu = status["directions"][d]["gpu"]
                finish(d, False)
                if queue:
                    start(queue.pop(0), gpu)
        if time.time() - T0 > DEADLINE_S:
            for d, (p, _, _) in list(procs.items()):
                log(f"deadline: stopping {d}")
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(120)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
                finish(d, True)
            for d in queue:
                status["directions"][d] = dict(state="not_started", reason="deadline")
            queue = []
            save()

    # -- benchmark (reference only used here, after translation) ---------------------------------
    score_json = W / f"scores_{VARIANT}.json"
    with open(W / f"scores_{VARIANT}.txt", "w") as out:
        rc = subprocess.run([sys.executable, str(bench / "eval_markdown.py"), "--csv", str(csv_path),
                             "--direction", "both", "--hyp", str(hyp), "--allow-missing",
                             "--save", str(score_json)], cwd=bench, stdout=out, stderr=subprocess.STDOUT).returncode
    status["benchmark"] = dict(returncode=rc, scores=str(score_json.name),
                               note="--allow-missing: a missing prediction is scored as an empty translation")
    log((W / f"scores_{VARIANT}.txt").read_text()[-4000:])
    status["utc_finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()

    # -- one archive for download ------------------------------------------------------------------
    with zipfile.ZipFile(W / f"results_{VARIANT}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in [hyp, *(W / f"work_{VARIANT}_{d}" for d in DIRECTIONS)]:
            if p.exists():
                for f in p.rglob("*"):
                    if f.is_file():
                        z.write(f, str(f.relative_to(W)))
        for f in W.glob(f"*_{VARIANT}*.*"):
            if f.suffix in (".json", ".txt", ".log"):
                z.write(f, f.name)
        z.write(W / "kernel_status.json", "kernel_status.json")
    shutil.rmtree(code, ignore_errors=True)
    shutil.rmtree(src_md, ignore_errors=True)
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

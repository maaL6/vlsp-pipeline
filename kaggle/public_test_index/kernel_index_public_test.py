"""Kaggle: pipeline-v1 phương án B (gom node, không marker) với Index-Translate-2B + adapter legal-SFT
pilot (Trang) trên public test BTC, hai chiều, rồi chấm bằng tools/score_public_test.py.

Input: một dataset chứa thư mục code `vlsp-pipeline/` (branch index2b-pilot) và `adapter/` (adapter_config.json +
adapter_model.safetensors của step 374). Base model tải từ Hugging Face đúng revision (Internet ON).
T4 x2: en-vi trên GPU 0, vi-en trên GPU 1. Dừng ở 11 h 15 m; work dir giữ cache, version sau mount output
của version trước (có work_index_<chiều>/) thì chạy tiếp.

USE_ADAPTER = False chạy Index gốc (đối chứng), run_id khác để không lẫn cache.
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

USE_ADAPTER = True
MODEL_ID = "IndexTeam/Index-Translate-2B"
REVISION = "a516854233b170140b57d36b48d7148c0edebabb"
MODEL_SHA256 = "2f499211fe352913464be69365e2e5724f913051efb3ae2b6fd0fa617a4c0b60"
ADAPTER_SHA256 = "8e327b833e4f88b64de8e71a8e5ad8d5659ca57982b685729492f59cb5147d62"   # pilot step 374
CSV_MD5 = {"fb14fdde3a060fe03d3e5266b01e5cab": "repo (LF)", "d86e055d3f84436ae07e6c27fe686bcc": "BTC gốc (CRLF)"}
TAG = "index_pilot374" if USE_ADAPTER else "index_base"
RUN_ID = f"index2b_structure_newline_{TAG}__pt50__kaggleT4"
CONFIG = "configs/index2b_pilot_newline.json"
BATCH_SIZE = int(os.environ.get("DOCMT_BATCH_SIZE", "8"))
DEADLINE_S = 11 * 3600 + 15 * 60
DIRECTIONS = ("en-vi", "vi-en")
W = Path("/kaggle/working")
T0 = time.time()
B_SCORES = {"en-vi": {"bleu": 68.90, "chrf": 83.39}, "vi-en": {"bleu": 46.62, "chrf": 79.17}}  # HY A1 B, 04/10


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
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.18.0", "peft==0.21.2", "tokenizers>=0.21",
                    "accelerate", "safetensors", "sacrebleu==2.6.0", "flash-linear-attention"], check=True)

    # -- code -----------------------------------------------------------------------------------
    src_code = find_one("/kaggle/input/**/configs/index2b_pilot_newline.json").parent.parent
    pipe = W / "pipeline"
    if pipe.exists():
        shutil.rmtree(pipe)
    shutil.copytree(src_code, pipe, ignore=shutil.ignore_patterns("results", ".git", "__pycache__"))
    csv_path = pipe / "dataset" / "public_test" / "en-vi-public_test.csv"
    csv_md5 = hashlib.md5(csv_path.read_bytes()).hexdigest()
    assert csv_md5 in CSV_MD5, f"public test CSV khác: {csv_md5}"

    # -- model + adapter ------------------------------------------------------------------------
    from huggingface_hub import snapshot_download
    model = Path(snapshot_download(MODEL_ID, revision=REVISION, cache_dir=str(W / "hf_cache")))
    assert sha256_file(model / "model.safetensors") == MODEL_SHA256, "Index weights khác revision đã chốt"
    adapter = ""
    if USE_ADAPTER:
        adapter = str(find_one("/kaggle/input/**/adapter_config.json").parent)
        assert sha256_file(Path(adapter) / "adapter_model.safetensors") == ADAPTER_SHA256, "adapter khác step 374"
    cfg = json.loads((pipe / CONFIG).read_text(encoding="utf-8"))
    cfg["extra"]["adapter"] = adapter
    cfg_path = W / f"config_{TAG}.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    import torch
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "không có GPU (bật GPU T4 x2)"
    log(f"tag={TAG} gpus={n_gpu} model={model} adapter={adapter or '-'} csv={CSV_MD5[csv_md5]}")

    # -- public test -> .md (CRLF -> LF, khớp nguồn run B của Lâm) --------------------------------
    src_md = W / "public_test_md"
    subprocess.run([sys.executable, str(pipe / "tools" / "export_public_test_md.py"), "--csv", str(csv_path),
                    "--out", str(src_md)], check=True)

    hyp = W / f"hyp_{TAG}"                # hyp/vi = chiều en-vi, hyp/en = chiều vi-en
    status = dict(tag=TAG, run_id=RUN_ID, config=CONFIG, batch_size=BATCH_SIZE, model=MODEL_ID, revision=REVISION,
                  model_sha256=MODEL_SHA256, adapter_sha256=ADAPTER_SHA256 if USE_ADAPTER else None,
                  gpu_count=n_gpu, gpu_name=torch.cuda.get_device_name(0), reference_used_by_pipeline=False,
                  utc_started=datetime.datetime.now(datetime.timezone.utc).isoformat(), directions={})

    def save():
        (W / "kernel_status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False))

    for d in DIRECTIONS:   # resume từ output version trước
        work = W / f"work_{TAG}_{d}"
        prev = sorted(glob.glob(f"/kaggle/input/**/work_{TAG}_{d}/logs/cache.jsonl", recursive=True))
        if prev and not work.exists():
            shutil.copytree(Path(prev[-1]).parent.parent, work)
            log(f"resume {d}: {Path(prev[-1]).parent.parent}")

    procs = {}

    def start(d, gpu):
        src, tgt = d.split("-")
        cmd = [sys.executable, "-m", "docmt", "run-dir", "--input-dir", str(src_md / src),
               "--output-dir", str(hyp / tgt), "--work-dir", str(W / f"work_{TAG}_{d}"),
               "--direction", d, "--config", str(cfg_path), "--run-id", RUN_ID,
               "--backend", "hf", "--model", str(model), "--tokenizer", str(model / "tokenizer.json"),
               "--set", f"batch_size={BATCH_SIZE}"]
        env = dict(os.environ, PYTHONPATH=str(pipe), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                   CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false")
        logf = open(W / f"run_{TAG}_{d}.log", "w")
        p = subprocess.Popen(cmd, cwd=pipe, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        procs[d] = (p, logf, time.time())
        status["directions"][d] = dict(gpu=gpu, state="running")
        save()
        log(f"start {d} on GPU {gpu}")

    def finish(d, timed_out):
        p, logf, t = procs.pop(d)
        logf.close()
        work = W / f"work_{TAG}_{d}"
        tgt = d.split("-")[1]
        n_pred = 0
        pred_dir = work / "predictions" / d
        for f in pred_dir.rglob("*.md") if pred_dir.exists() else []:
            dst = hyp / tgt / f.relative_to(pred_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
            n_pred += 1
        docs = [json.loads(x) for x in (work / "logs/document_status.jsonl").read_text().splitlines()] \
            if (work / "logs/document_status.jsonl").exists() else []
        rj_path = work / "logs" / f"{d}.run.json"
        rj = json.loads(rj_path.read_text()) if rj_path.exists() else {}
        keep = ("documents", "nodes", "units", "requests", "requests_sent", "cache_hits", "recovery_requests",
                "retry_count", "runtime", "prompt_tokens", "generated_tokens", "peak_vram", "oom_count",
                "op_error_count", "unresolved_units", "unresolved_empty_nodes", "parameter_count")
        st = {}
        for r in docs:
            st[r.get("status")] = st.get(r.get("status"), 0) + 1
        status["directions"][d].update(state="timeout" if timed_out else "finished", returncode=p.returncode,
                                       seconds=round(time.time() - t, 1), predictions=n_pred, expected=50,
                                       document_status=st, run={k: rj.get(k) for k in keep})
        save()
        log(f"end {d}: {status['directions'][d]}")

    queue = list(DIRECTIONS)
    for g in range(min(n_gpu, 2)):
        if queue:
            start(queue.pop(0), g)
    last_report = 0
    while procs:
        time.sleep(30)
        if time.time() - last_report > 600:      # tiến độ mỗi 10 phút
            last_report = time.time()
            for d in procs:
                f = W / f"work_{TAG}_{d}" / "logs" / "document_status.jsonl"
                n = len(f.read_text().splitlines()) if f.exists() else 0
                log(f"{d}: {n}/50 tài liệu xong")
        for d, (p, _, _) in list(procs.items()):
            if p.poll() is not None:
                gpu = status["directions"][d]["gpu"]
                finish(d, False)
                if queue:
                    start(queue.pop(0), gpu)
        if time.time() - T0 > DEADLINE_S:
            for d, (p, _, _) in list(procs.items()):
                log(f"deadline: dừng {d}")
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

    # -- chấm (reference chỉ dùng ở đây, sau khi dịch) -------------------------------------------
    score_json = W / f"scores_{TAG}.json"
    r = subprocess.run([sys.executable, str(pipe / "tools" / "score_public_test.py"), "--csv", str(csv_path),
                        "--hyp", str(hyp), "--save", str(score_json)], capture_output=True, text=True)
    (W / f"scores_{TAG}.txt").write_text(r.stdout + r.stderr)
    log(r.stdout + r.stderr[-2000:])
    if score_json.exists():
        sc = json.loads(score_json.read_text())
        status["scores"] = {d: {k: sc[d][k] for k in ("bleu", "chrf", "chrf_pp", "bp", "ratio")} | {"missing": len(sc[d]["missing"])}
                            for d in DIRECTIONS}
        status["vs_hy_a1_B"] = {d: {"bleu_delta": round(sc[d]["bleu"] - B_SCORES[d]["bleu"], 2),
                                    "chrf_delta": round(sc[d]["chrf"] - B_SCORES[d]["chrf"], 2)} for d in DIRECTIONS}
        log(f"so với HY A1 phương án B: {status['vs_hy_a1_B']}")
    status["utc_finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()

    with zipfile.ZipFile(W / f"results_{TAG}.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in [hyp, *(W / f"work_{TAG}_{d}" for d in DIRECTIONS)]:
            if p.exists():
                for f in p.rglob("*"):
                    if f.is_file():
                        z.write(f, str(f.relative_to(W)))
        for f in W.glob(f"*_{TAG}*.*"):
            if f.suffix in (".json", ".txt", ".log"):
                z.write(f, f.name)
        z.write(W / "kernel_status.json", "kernel_status.json")
    shutil.rmtree(pipe, ignore_errors=True)
    shutil.rmtree(src_md, ignore_errors=True)
    shutil.rmtree(W / "hf_cache", ignore_errors=True)
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

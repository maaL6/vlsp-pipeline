"""Validate downloaded evidence and summarize GPU measurements against historical B."""
import json
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    out = Path(sys.argv[1]).resolve()
    archive = out / "results_v13.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                dst = (out / name).resolve()
                if not dst.is_relative_to(out):
                    raise ValueError("unsafe archive member")
            z.extractall(out)
    status = json.loads((out / "kernel_status.json").read_text(encoding="utf-8"))
    if all(status.get("benchmark", {}).get(v, {}).get("complete_50x2") for v in ("batch_length", "glossary")) \
            and not all((out / f"scores_local_{v}.json").exists() for v in ("batch_length", "glossary")):
        scored = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "kaggle/public_test/evaluate_v13.py"), str(out)],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        (out / "local_rescore.log").write_text(scored.stdout + scored.stderr, encoding="utf-8")
        if scored.returncode:
            print("Local rescore failed; see local_rescore.log. Preserving Kaggle scores.")
    baseline = json.loads((ROOT / "reports/public_test_abc_20261004.json").read_text(encoding="utf-8"))
    oldstatus = json.loads((ROOT / "results/public_test_20261004/B_gom_node_khong_marker/kernel_status.json").read_text(encoding="utf-8"))
    report = {"status": "completed", "kernel": "wann179/vlsp26-v13-batching-glossary-pt50",
              "evidence": str(out), "gpu_names": status["gpu_names"], "gpu_map": status["gpu_map"],
              "csv_sha256": status["csv_sha256"], "model_sha256": status["model_sha256"],
              "bundle_manifest_sha256": status["bundle_manifest_sha256"], "runs": {},
              "comparison_note": "Historical B v1.1 GPU runtime, not a fresh v1.3 FIFO no-glossary control. v1.2 runtime was replay only."}
    report["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest() if archive.exists() else None
    lines = ["# Benchmark v1.3 Kaggle", "", "Số đo GPU thật; GPU 0 batching, GPU 1 glossary. Batch size 4.", "",
             "| Bản | Chiều | Doc | BLEU raw | Δ BLEU so B v1.1 | Giờ dịch | Recovery | Unresolved | Padding |",
             "|---|---|---|---|---|---|---|---|---|"]
    for variant in ("batch_length", "glossary"):
        scores = json.loads((out / f"scores_{variant}.json").read_text(encoding="utf-8")) if (out / f"scores_{variant}.json").exists() else []
        local_file = out / f"scores_local_{variant}.json"
        local_scores = json.loads(local_file.read_text(encoding="utf-8")) if local_file.exists() else []
        for direction in ("en-vi", "vi-en"):
            key = f"{variant}/{direction}"
            st = status["runs"][key]
            run = st.get("run", {})
            files = list((out / f"hyp_{variant}" / direction.split("-")[1]).rglob("*.md"))
            score = next((s for s in scores if f"[{direction}]" in s["system"]), None)
            local = next((s for s in local_scores if f"[{direction}]" in s["system"]), None)
            if local is not None and score is not None:
                assert all(local[k] == score[k] for k in ("raw", "nfc", "text", "missing")), "local/Kaggle score mismatch"
            complete = st["state"] == "completed" and st.get("predictions") == 50 and len(files) == 50 and score is not None and score["missing"] == 0
            if not complete:
                report["status"] = "incomplete"
            metrics = run.get("backend", {}).get("batch_metrics", {})
            padded = metrics.get("padded_prompt_tokens", 0)
            pad_fraction = 1 - metrics.get("prompt_tokens", 0) / padded if padded else None
            bleu = score["raw"]["bleu"] if score else None
            delta = round(bleu - baseline["scores"]["B"][direction]["raw"]["bleu"], 2) if bleu is not None else None
            runtime = run.get("runtime")
            oldtime = oldstatus["directions"][direction]["run"].get("runtime")
            rec = {"complete_50": complete, "predictions": len(files), "score": score,
                   "local_rescore_verified": local is not None,
                   "bleu_delta_vs_B_v11": delta, "runtime_seconds": runtime,
                   "wall_seconds": st.get("wall_seconds"), "historical_B_seconds": oldtime,
                   "historical_speedup": oldtime / runtime if oldtime and runtime else None,
                   "padding_fraction": pad_fraction, "batch_metrics": metrics,
                   **{k: run.get(k) for k in ("requests", "requests_sent", "cache_hits", "recovery_requests",
                                               "retry_count", "unresolved_units", "unresolved_empty_nodes",
                                               "oom_count", "op_error_count", "prompt_tokens", "generated_tokens")}}
            report["runs"][key] = rec
            hours = f"{runtime/3600:.3f}" if runtime else "—"
            padding = f"{pad_fraction:.1%}" if pad_fraction is not None else "—"
            lines.append(f"| {variant} | {direction} | {len(files)}/50 | {bleu} | {delta} | {hours} | {rec['recovery_requests']} | {rec['unresolved_units']} | {padding} |")
    lines += ["", f"Trạng thái: **{report['status']}**. Chỉ coi benchmark đủ khi cả 4 lượt có 50 dự đoán và không thiếu file lúc chấm.",
              "", "Thời gian so sánh lấy từ B v1.1 chạy GPU trước đây; B v1.2 chỉ có replay, chưa đo GPU. Chưa chạy một baseline v1.3 mới cùng phiên nên chênh lệch tốc độ mang tính tham khảo.",
              "", "Padding là phần token đệm trong tensor đầu vào. Baseline cũ chưa ghi chỉ số này nên không có mức giảm padding đối chứng.",
              "", f"Model SHA256: `{report['model_sha256']}`. CSV SHA256: `{report['csv_sha256']}`.",
              "", f"Bundle manifest SHA256: `{report['bundle_manifest_sha256']}`. Key không nằm trong bundle hay báo cáo."]
    report["all_local_rescores_verified"] = all(r["local_rescore_verified"] for r in report["runs"].values())
    if report["all_local_rescores_verified"]:
        lines += ["", "Đã chấm lại local bằng evaluator chuẩn: cả 4 lượt khớp Kaggle ở BLEU/chrF raw, NFC và text; không thiếu file."]
    (ROOT / "reports/v13_kaggle_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "reports/v13_kaggle_run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

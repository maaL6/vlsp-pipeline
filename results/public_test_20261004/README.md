# Public test BTC — kết quả A / B / C (Kaggle, 03–04/10/2026)

Full pipeline-v1 trên public test của BTC (50 tài liệu, 2 chiều en-vi và vi-en), chấm bằng
`benchmark/eval_markdown.py` (SacreBLEU 2.6.0, `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp`).
Bản tham chiếu chỉ dùng khi chấm, pipeline không đọc nó.

| Phương án | Mô tả | Config | Kernel Kaggle |
|---|---|---|---|
| **A** | gom node + marker `[SEG_000123]` | `configs/hy_a1_structure.json` | `wann179/vlsp26-lam-pt50-markers` |
| **B** (đã chốt) | gom node + không marker (nối bằng xuống dòng) | `configs/hy_a1_structure_newline.json` | `maal06/vlsp26-lam-pt50-newline` |
| **C** | không gom node + không marker (mỗi node một request) | `configs/hy_a1_structure_nomarker.json` | `wann179/vlsp26-lam-pt50-nomarker` |

Chung: model A1 merged (`model.safetensors` sha256 `5dc305ae…`), backend HF fp16, greedy,
`repetition_penalty=1.05`, `batch_size=4`, Kaggle T4 ×2 (mỗi chiều một GPU), CSV md5 `d86e055d…`.

## Điểm (raw, không chuẩn hoá)

| | en-vi BLEU | en-vi chrF | vi-en BLEU | vi-en chrF | Thiếu output |
|---|---|---|---|---|---|
| A gom node + marker | 68.70 | 83.24 | 46.37 | 79.01 | 0 |
| **B gom node + không marker** | **68.90** | **83.39** | **46.62** | **79.17** | 0 |
| C không gom + không marker | 67.69 | 82.64 | 45.23 | 78.23 | 0 |

Bootstrap theo tài liệu (1000 lần, khoảng 95%): B−A = +0.20 [+0.05, +0.37] en-vi,
+0.25 [−0.05, +0.65] vi-en (vi-en chưa chắc chắn); B−C = +1.21 / +1.39 (rõ ràng).
Chi tiết: `reports/public_test_abc_20261004.json`.

## Thời gian và trạng thái tài liệu

| | en-vi (giờ) | vi-en (giờ) | Request en-vi / vi-en | `completed` / `completed_with_unresolved` en-vi | vi-en |
|---|---|---|---|---|---|
| A | 5.53 | 3.10 | 7347 / 4078 | 48 / 2 | 48 / 2 |
| B | 4.57 | 2.89 | 5220 / 5295 | 44 / 6 | 49 / 1 |
| C | 4.35 | 2.90 | 12581 / 12537 | 42 / 8 | 49 / 1 |

Mỗi phương án chạy một lần trên máy Kaggle khác nhau: chênh lệch tốc độ dưới khoảng 5% nằm trong nhiễu.

## Lưu ý

- Output B ở đây là **bản v1.1** (code lúc chạy: `bf21111` + thay đổi newline chưa commit, sau đó commit thành
  `2a1f852`). Bản v1.2 (`c62f442`, unit gom lỗi luôn dịch lại từng node) **chưa chạy GPU**; mô phỏng bằng replay
  trace B + C: BLEU 68.90 / 46.61, unit chưa xử lý được 7+1 → 2+0
  (`reports/public_test_b_v12_replay_20261004.json`).
- `Quyen_dan_su/390` chỉ khoảng 1.5 BLEU ở cả ba phương án: bản tham chiếu tiếng Việt dài gấp khoảng 2,3 lần
  nguồn tiếng Anh (lệch nguồn/tham chiếu, không phải lỗi pipeline).

## Nội dung thư mục

```
scores_compare.{json,txt}        bảng so sánh 3 phương án (chấm lại trên máy local)
<phương án>/
  predictions/vi/…  predictions/en/…   bản dịch .md (vi = chiều en-vi, en = chiều vi-en), 50 + 50 file
  scores.{json,txt}                   điểm do kernel chấm (có điểm từng tài liệu)
  kernel_status.json                  config, hash model/bundle, thời gian, trạng thái tài liệu
  logs/kernel.log, logs/run_<chiều>.log
  logs/<chiều>.run.json               thống kê run (request, recovery, token, VRAM…)
  logs/document_status_<chiều>.jsonl  trạng thái từng tài liệu
  full_work_<variant>.zip             toàn bộ output kernel: work dir gồm generation_trace.jsonl,
                                      prompts.jsonl, cache.jsonl — dùng cho ReplayBackend
```

Chấm lại: `python ../benchmark/eval_markdown.py --direction both --hyp results/public_test_20261004/*/predictions`
(chạy trong `pipeline-v1/`; `benchmark/` nằm cạnh `pipeline-v1/` và cần `benchmark/data/en-vi-public_test.csv`).
Replay không cần GPU: giải nén `full_work_<variant>.zip`, dùng `docmt/backends/replay.py` (ví dụ `tools/replay_equivalence.py`) với trace trong `work_*/logs/`.

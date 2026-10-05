# kaggle/public_test — full public test trên Kaggle

Chạy full pipeline-v1 (HF + A1 merged) trên public test BTC (50 doc, en-vi + vi-en), rồi chấm bằng
`benchmark/eval_markdown.py`. Dùng để so các phương án (A/B/C, xem README gốc › *Kết quả public test*).

| File | Tác dụng |
|---|---|
| `kernel_public_test.py` | Script chạy trong kernel Kaggle. `VARIANT` được `push.sh` điền: `markers` (A), `newline` (B), `nomarker` (C). Kiểm hash bundle, model (`5dc305ae…`), md5 CSV; xuất `.md`; chạy en-vi trên GPU 0 và vi-en trên GPU 1; dừng ở 11 h 15 m (giữ cache để chạy tiếp); chấm `--allow-missing`; nén `results_<variant>.zip` |
| `push.sh` | Đóng gói `pipeline-v1` + `benchmark/` thành dataset private `<OWNER>/vlsp26-lam-pipeline-v1-pt50`, push kernel private `<OWNER>/vlsp26-lam-pt50-<variant>` (GPU T4) |
| `fetch_and_compare.sh` | Tải `results_<variant>.zip` + `kernel_status.json` của các kernel, in bảng BLEU/chrF các phương án cạnh nhau |

```bash
# Kaggle CLI >= 1.8 (đọc KAGGLE_API_TOKEN). Tài khoản phải có dataset model <OWNER>/vlsp26-hy-a1-merged.
KAGGLE_API_TOKEN=... OWNER=maal06 ./push.sh newline          # B; hoặc markers | nomarker | all (= A + C)
KAGGLE_API_TOKEN=... OWNER=maal06 RESUME=1 ./push.sh newline # chạy tiếp sau khi kernel chạm 11 h 15 m

# Tải và chấm; KERNELS = "owner:variant[:TÊN_BIẾN_TOKEN]"
KERNELS="maal06:newline" KAGGLE_API_TOKEN=... ./fetch_and_compare.sh results
```

Token chỉ đọc từ biến môi trường, không ghi ra file. `build/` (bundle tạm) và `results/` (kết quả tải về)
bị `.gitignore`; tóm tắt kết quả 04/10/2026 được giữ ở `reports/public_test_abc_20261004.json`.

Run 04/10/2026: A `wann179/vlsp26-lam-pt50-markers`, C `wann179/vlsp26-lam-pt50-nomarker`,
B `maal06/vlsp26-lam-pt50-newline`; run_id `hy_a1_structure{,_nomarker,_newline}__pt50__kaggleT4__20261004`.

## v1.3: hai thay đổi riêng trên cùng máy T4 ×2

`kernel_v13.py` chạy `hy_a1_v13_batch_length.json` trên GPU 0 và
`hy_a1_v13_glossary.json` trên GPU 1. Mỗi GPU chạy EN→VI rồi VI→EN,
chuyển chiều độc lập; batch size 4, fp16, greedy, giữ delimiter newline.
Kernel bắt buộc máy có đúng hai T4, kiểm SHA256 code/model/CSV trước khi dịch.

- `build_v13_bundle.py`: đóng gói working tree chưa commit và evaluator chuẩn trong `build/original/`.
- `preflight_v13.py`: kiểm tra đủ 50 doc × 2 chiều × 2 config bằng model giả và tokenizer A1 thật;
  báo cáo `reports/v13_mock_preflight.json`, không chấm BLEU cho bản echo.
- `monitor_v13.py`: nhận token qua prompt ẩn, giữ trong bộ nhớ, theo dõi và tải outputs khi xong.
- `evaluate_v13.py <output-dir>`: chấm lại local bằng evaluator gốc; xử lý giới hạn CSV của Windows.
- `summarize_v13.py <output-dir>`: kiểm tra đủ predictions, tóm tắt BLEU/runtime/recovery/unresolved/padding;
  tạo `reports/v13_kaggle_results.json` và cập nhật `reports/v13_kaggle_run.md`.

Kernel private: `wann179/vlsp26-v13-batching-glossary-pt50`.
Dataset code private: `wann179/vlsp26-pipeline-v13-pt50`.
Theo dõi trạng thái hiện tại trong `reports/v13_kaggle_run.md`.

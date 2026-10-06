# kaggle/public_test_index — Index-Translate-2B + adapter pilot trên public test

Branch `index2b-pilot` (Trang, 06/10/2026). Chạy pipeline-v1 **phương án B** (gom node, không marker, cùng cấu trúc/recovery
như HY) nhưng model là `IndexTeam/Index-Translate-2B@a5168542` + adapter LoRA legal-SFT pilot step 374, rồi chấm.

## Thay đổi trong code (đường HY không đổi)

| File | Thay đổi |
|---|---|
| `docmt/steps/s04_prompt_index.py` | Mới: prompt native Index (giống prompt dùng khi train adapter); direct hoặc instTrans |
| `docmt/steps/s04_prompt.py`, `s05_budget.py` | Dùng prompt Index khi `cfg.extra.model_family == "index"`; mặc định `"hy"` |
| `docmt/backends/hf.py` | `cfg.extra.adapter`: nạp LoRA bằng peft rồi `merge_and_unload`; sha256 adapter + model_family vào `decode_config` (khoá cache) chỉ khi có; bỏ khối `<think>` với Index |
| `configs/index2b_pilot_newline.json` | = `hy_a1_structure_newline.json` + `model_family=index`, `repetition_penalty=1.0` (native Index), owner/branch |
| `tools/export_public_test_md.py`, `tools/score_public_test.py` | Thay `benchmark/` (không có trong repo): xuất nguồn khớp sha nguồn run B 50/50; chấm lại B ra đúng 68.90 / 46.62 |

Đã kiểm (CPU): HY mock trên 100 file public test giống `main` từng byte (output, prompt, khoá cache, trace); prompt Index
== chat template của tokenizer Index (4/4 dạng); Index mock 100/100 completed; HF backend + LoRA trên model tí hon chạy hết.
Chưa kiểm trên GPU trước lượt này.

## Chạy

Dataset Kaggle chứa `vlsp-pipeline/` (branch này, không cần `results/`) và `adapter/` (step 374). Notebook
`kaggle/public_test_index/run_index_public_test.ipynb`, GPU T4 x2, Internet ON, Save & Run All.
`USE_ADAPTER = False` trong kernel để chạy Index gốc làm đối chứng (run_id/cache riêng).
Output: `results_index_pilot374.zip` (bản dịch `hyp_*/vi|en`, work dir có trace/cache để resume), `scores_*.json`,
`kernel_status.json` (thời gian, trạng thái tài liệu, điểm, chênh lệch so với HY A1 B).

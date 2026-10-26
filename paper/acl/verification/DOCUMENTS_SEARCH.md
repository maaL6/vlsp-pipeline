# Tìm nguồn kết quả trong Documents — 2026-10-27

## Kết luận

Chưa tìm được log/output chấm BLEU gốc khớp cặp Index **62.59 / 41.76**, hoặc các dòng baseline trong bảng chính. Cặp Index này có trong bản thảo, bản PDF lưu cũ và lịch sử nhập paper; những bản sao đó không phải bằng chứng thực nghiệm độc lập.

## Phạm vi đã kiểm tra

- Toàn cây `C:/Users/trand/Documents`: 6,353 file văn bản/log/report/notebook được tìm theo các giá trị đích, gồm cả JSONL, HTML và file ẩn ngoài các thư mục dependency/Git internals/model binary.
- Có kiểm tra biến thể số thập phân chưa làm tròn, dấu phẩy và thang điểm 0–1. Các trùng số là thời gian chạy, n-gram precision theo tài liệu hoặc dữ liệu không liên quan đã bị loại.
- Quét bổ sung 1,223 file metadata/log/notebook cho cặp baseline Index 43.96/31.22 và các giá trị baseline tương ứng.
- Đọc metadata/log/report trong 20 ZIP liên quan VLSP/Index/Hy-MT; không giải nén hoặc thay đổi archive.
- Đọc 143 phiên bản blob report/score/paper liên quan trong lịch sử Git của hai checkout pipeline. Checkout chính có 38 commits reachable tại thời điểm kiểm tra.
- Một thư mục `.pytest_cache` của checkout cũ bị từ chối truy cập khi liệt kê; không có file nội dung được chọn để quét bị lỗi đọc. Không tìm trong weights/binary hoặc tái chạy toàn bộ inference.

## Những run Index tìm thấy

| Run | EN→VI BLEU | VI→EN BLEU | Bằng chứng |
|---|---:|---:|---|
| v3.3 archive | 67.58 | 46.20 | `C:/Users/trand/Documents/Index-Translate-2B-Legal-LoRA-v33-FULL/results/public_test_scores_v33.json` |
| Release cũ | 63.95 | 43.42 | `C:/Users/trand/Documents/Index-Translate-2B-Legal-LoRA-v33-FULL/results/public_test_scores_release_old.json` |
| v3.3 glossary ablation | 67.69 | 45.86 | `scores.json` bên trong `C:/Users/trand/Documents/vlsp-pipeline-main/kaggle/index_v33_ablation/results/glossary/output/results_index_public100.zip` |
| Pilot step 374 | 60.05 | 41.10 | Commit `f2a5591`, `results/public_test_20261006_index_pilot374/scores_index_pilot374.json` |
| Dòng trong paper | 62.59 | 41.76 | Chưa có log/output gốc khớp |

Không tự thay điểm paper bằng điểm của các run khác; checkpoint, dữ liệu huấn luyện và inference config có thể khác nhau.

## Nguồn sớm nhất của con số trong bản thảo

`git log --all -S '62.59' -- '*.tex'` tìm thấy commit `97e5442` (2026-10-15 01:13:44 +07:00, “add IEEE paper folder”). Bản lưu `out/paper-cleanup/paper_en_20261015.pdf` cũng có đúng cặp số này. Như vậy các con số đã có trong bản thảo trước đợt chỉnh ACL hiện tại; chưa xác định được log thực nghiệm tạo ra chúng.

Có thể đọc lại run pilot đã bị xóa khỏi checkout bằng:

```powershell
git show f2a5591:results/public_test_20261006_index_pilot374/scores_index_pilot374.json
git show f2a5591:results/public_test_20261006_index_pilot374/README.md
```

## Cách xử lý tiếp

Giữ trạng thái chưa xác minh cho các dòng này. Để chứng thực, cần log/output đúng run hoặc tái chấm predictions đúng checkpoint/config. Tái chạy một cấu hình khác sẽ tạo kết quả mới, không xác minh nguồn của con số cũ. Lượt tìm kiếm này không sửa manuscript, PDF hay các số liệu trong bảng.

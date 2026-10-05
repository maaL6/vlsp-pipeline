# Output benchmark v1.3 trên Kaggle

Kernel: `wann179/vlsp26-v13-batching-glossary-pt50`, version 1.
Chạy ngày 04/10/2026, hoàn thành; Tesla T4 ×2, batch size 4.
GPU 0 chạy length batching; GPU 1 chạy glossary với batching FIFO.
Mỗi GPU chạy EN→VI rồi VI→EN, 50 tài liệu mỗi chiều.

- `hyp_batch_length/`, `hyp_glossary/`: đủ 200 file dự đoán Markdown.
- `scores_*.json`, `scores_*.txt`: điểm Kaggle và điểm chấm lại local.
- `kernel_status.json`: trạng thái hoàn thành và thống kê GPU thật.
- `results_v13.zip`: output đầy đủ, gồm work dirs, generation traces, prompts,
  cache, document status và predictions. Work dirs giải nén không lưu trùng trong Git.
- `run_*.log`, `local_rescore.log`: log chạy và chấm lại.

Cả bốn lượt chấm local khớp Kaggle ở BLEU/chrF raw, NFC và text, missing=0.
Báo cáo đánh giá: `reports/v13_detailed_assessment.md`.
Baseline đối chiếu: B v1.1 lịch sử trong `reports/public_test_abc_20261004.json`.
Không có baseline v1.3 FIFO/no-glossary chạy mới cùng phiên.
Credentials không nằm trong output được lưu.

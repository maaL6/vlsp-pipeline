# Benchmark v1.3 Kaggle

Số đo GPU thật; GPU 0 batching, GPU 1 glossary. Batch size 4.

| Bản | Chiều | Doc | BLEU raw | Δ BLEU so B v1.1 | Giờ dịch | Recovery | Unresolved | Padding |
|---|---|---|---|---|---|---|---|---|
| batch_length | en-vi | 50/50 | 68.89 | -0.01 | 3.591 | 3643 | 2 | 4.5% |
| batch_length | vi-en | 50/50 | 46.65 | 0.03 | 2.268 | 3857 | 0 | 3.6% |
| glossary | en-vi | 50/50 | 68.97 | 0.07 | 4.697 | 3954 | 3 | 29.8% |
| glossary | vi-en | 50/50 | 46.26 | -0.36 | 3.001 | 4423 | 0 | 29.6% |

Trạng thái: **completed**. Chỉ coi benchmark đủ khi cả 4 lượt có 50 dự đoán và không thiếu file lúc chấm.

Thời gian so sánh lấy từ B v1.1 chạy GPU trước đây; B v1.2 chỉ có replay, chưa đo GPU. Chưa chạy một baseline v1.3 mới cùng phiên nên chênh lệch tốc độ mang tính tham khảo.

Padding là phần token đệm trong tensor đầu vào. Baseline cũ chưa ghi chỉ số này nên không có mức giảm padding đối chứng.

Model SHA256: `5dc305ae6c33c15a8440a4a8f9183e411004079eb737bc2da417b6a932dead81`. CSV SHA256: `dd7e0070a735da6a82de1c5c426afb8fdc4a9b0bf16c1271cf2396ddb95d4ef3`.

Bundle manifest SHA256: `5f297e30a966c73abea66c631d6c09e513ef90f3084e7e5bd356501aef2b417f`. Key không nằm trong bundle hay báo cáo.

Đã chấm lại local bằng evaluator chuẩn: cả 4 lượt khớp Kaggle ở BLEU/chrF raw, NFC và text; không thiếu file.

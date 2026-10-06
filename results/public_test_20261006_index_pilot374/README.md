# Public test — Index-Translate-2B + adapter pilot step 374 (Kaggle, 06/10/2026)

Pipeline-v1 phương án B (gom node, không marker; cùng parser/pack/recovery với HY B), model
`IndexTeam/Index-Translate-2B@a5168542` + adapter LoRA legal-SFT pilot step 374 (sha256 `8e327b83…`, train trên
payload v2 của Linh, 11.988 ID chung HY), prompt native Index, greedy, `repetition_penalty=1.0`, fp16, batch 8,
Kaggle T4 ×2 (en-vi GPU 0, vi-en GPU 1). Chấm `tools/score_public_test.py` (SacreBLEU 2.6.0, cùng số với
`eval_markdown.py` trên output B). Kernel: `kaggle/public_test_index/`. Reference chỉ dùng khi chấm.

## Điểm (raw)

| | en-vi BLEU | en-vi chrF | vi-en BLEU | vi-en chrF | Thời gian (en-vi / vi-en) |
|---|---|---|---|---|---|
| HY A1, phương án B (04/10) | 68.90 | 83.39 | 46.62 | 79.17 | 4,5 h / 2,9 h |
| **Index-2B + pilot 374, B** | **60.05** | **78.71** | **41.10** | **76.22** | 3,3 h / 2,3 h |
| Chênh | −8.85 | −4.68 | −5.52 | −2.95 | |

Tỷ lệ độ dài so với reference: en-vi 1.049 (HY B 0.943), vi-en 1.075 (HY B 1.037). 100/100 output, 0 OOM.
Trạng thái tài liệu: en-vi 46 `completed_with_unresolved` / 4 `completed` (176 unit unresolved; HY B 6/44, 7 unit);
vi-en 8 / 42 (13 unit).

## Nguyên nhân chính (đọc trace)

- **Nguồn rất ngắn → model không dừng.** Lượt đầu (unit gom) ít lỗi; khi unit lỗi `line_count_mismatch` được dịch lại
  từng node, node chỉ vài token (`No.`, `RESOLUTION`, ô bảng `other annual plants`) làm Index + adapter viết tiếp
  văn bản bịa hoặc lặp một cụm đến hết budget 256 (`no_eos`). en-vi: 300 request `no_eos` (226 ở lượt dịch lại,
  phần lớn unit `single`, trung vị 4 token nguồn); vi-en: 26.
- Khi bản đầu và bản dịch lại cùng lỗi, selection vẫn giữ một bản: **238 request `no_eos` (≈63k token) nằm trong
  output cuối en-vi**, ở 46/50 tài liệu → độ dài 1.049 và phần lớn mức tụt BLEU en-vi.
- vi-en ít bị (16 request). Bỏ 7 tài liệu có lỗi này, vi-en vẫn 41.5 BLEU → khoảng cách ~5 BLEU với HY B không
  chỉ do lỗi trên.

## Kết luận

Pilot **không đạt** điều kiện mở full SFT của draft (chrF++ cách A1 ≤ 1 điểm mỗi chiều) trên pipeline B ở cả hai chiều.
Chưa đo Index gốc (không adapter) trên cùng pipeline, nên chưa tách được phần do adapter và phần do backbone.
Chưa thử: budget theo độ dài cho nguồn ngắn, `repetition_penalty` 1.05, coi output dài bất thường so với nguồn là
hard failure. Không có review người.

## Nội dung

`predictions/vi` (chiều en-vi), `predictions/en` (vi-en); `scores_index_pilot374.{json,txt}` (điểm từng doc);
`kernel_status.json`; `config_index_pilot374.json`; `logs/document_status_*.jsonl`, `logs/*.run.json`.
Trace/cache đầy đủ (≈60 MB) không đưa vào git; ở máy Trang.

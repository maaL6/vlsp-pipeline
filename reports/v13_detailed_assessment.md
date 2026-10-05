# Đánh giá toàn diện v1.3: batching và glossary

## Kết luận

Nên chọn batching theo độ dài làm cấu hình mặc định tiếp theo. Thời gian dịch giảm khoảng 21% so với B v1.1 lịch sử, điểm BLEU/chrF gần như giữ nguyên. Chưa nên bật glossary hiện tại cho cả hai chiều: lợi ích EN→VI nhỏ và chưa chắc chắn, VI→EN giảm rõ hơn, đồng thời tốn thêm thời gian và recovery.

## Dữ liệu và độ tin cậy

Đã tải kết quả private kernel wann179/vlsp26-v13-batching-glossary-pt50 về results/v13_kaggle_20261004. Đủ 200/200 file: mỗi bản 50 tài liệu × hai chiều. Cả bốn lượt chấm lại local bằng evaluator chuẩn khớp Kaggle ở raw/NFC/text BLEU và chrF, missing=0. Đây là số đo GPU thật, không phải mock hay replay.

T4 ×2, GPU 0 batching; GPU 1 glossary; mỗi GPU EN→VI rồi VI→EN; batch size 4, cùng model và public CSV. Glossary có 368 cặp từ nguồn train, soft_terms=4, giữ batch FIFO, không dùng source context/history. Audit không thấy trùng sample ID train với dev/public. Co-occurrence trong train không bảo đảm mỗi cặp đúng mọi ngữ cảnh.

Baseline ban đầu là B v1.1 (gom node, newline, không glossary). B v1.2 là replay và có thay đổi recovery, chưa đo GPU. Không có baseline v1.3 FIFO/no-glossary chạy mới cùng phiên: tốc độ so lịch sử mang tính tham khảo; cải thiện unresolved không thể quy hoàn toàn cho batching hoặc glossary.

## Điểm dịch và tốc độ

| Bản | Chiều | BLEU raw | Δ BLEU | chrF | Giờ dịch | Δ thời gian vs B v1.1 |
|---|---|---:|---:|---:|---:|---:|
| B v1.1 | en-vi | 68.90 | — | 83.39 | 4.543 | — |
| batch_length | en-vi | 68.89 | -0.01 | 83.38 | 3.591 | -20.96% |
| glossary | en-vi | 68.97 | +0.07 | 83.43 | 4.697 | +3.39% |
| B v1.1 | vi-en | 46.62 | — | 79.17 | 2.870 | — |
| batch_length | vi-en | 46.65 | +0.03 | 79.18 | 2.268 | -20.98% |
| glossary | vi-en | 46.26 | -0.36 | 79.14 | 3.001 | +4.56% |

Batching tiết kiệm 57,1 phút EN→VI và 36,1 phút VI→EN; tổng 93,2 phút/GPU cho 100 tài liệu. Tổng thời gian dịch trên một GPU: baseline 7,413 giờ; batching 5,859 giờ; glossary 7,698 giờ. Throughput batching tương đương khoảng 1,265× baseline. Glossary tăng thời gian khoảng 3,4%/4,6%. Đây là runtime pipeline; thời gian toàn kernel còn gồm load model, chấm và đóng gói.

BLEU đo mức khớp với bản tham chiếu, chrF đo mức khớp ký tự. Chúng không trực tiếp bảo đảm đúng nghĩa pháp lý. NFC và điểm text cũng giữ cùng xu hướng: batching text 68,46/45,82 so baseline 68,48/45,79; glossary text 68,55/45,43. Sự giảm VI→EN không chỉ xuất hiện ở Markdown raw.

## Mức chắc chắn và từng tài liệu

Bootstrap ghép cặp: lấy lại 50 tài liệu có hoàn lại 2.000 lần, seed 42, tính lại corpus BLEU từ thống kê SacreBLEU 2.6.0. Khoảng dưới đây mô tả biến động theo tài liệu của tập này, không bao gồm biến động nhiều lần chạy GPU hoặc khái quát mọi dữ liệu.

| Bản | Chiều | Tăng / giảm / gần bằng | Khoảng 95% của Δ corpus BLEU |
|---|---|---:|---:|
| batch_length | en-vi | 14 / 24 / 12 | [-0.030; +0.009] |
| glossary | en-vi | 30 / 19 / 1 | [-0.007; +0.160] |
| batch_length | vi-en | 16 / 19 / 15 | [-0.022; +0.123] |
| glossary | vi-en | 18 / 32 / 0 | [-0.651; -0.110] |

Gần bằng nghĩa là chênh điểm từng tài liệu ≤0,005. Khoảng batching hai chiều và glossary EN→VI đều chứa 0; chưa có bằng chứng chắc chắn rằng điểm tăng/giảm. Glossary VI→EN có khoảng hoàn toàn âm. Số tài liệu tăng điểm không quyết định corpus BLEU: tài liệu dài đóng góp nhiều thống kê hơn.

Batching không bảo đảm output giống hệt: chỉ 8/50 file EN→VI và 11/50 VI→EN giống baseline theo nội dung Unicode đã đọc. Đổi thành phần batch có thể làm phép tính fp16 và chọn token/recovery khác đi; đây là khả năng giải thích, chưa được tách kiểm chứng. Chênh điểm vẫn nhỏ.

## Kết quả theo nhóm pháp luật

Mỗi ô: BLEU của baseline → batching → glossary. Đây là corpus BLEU tính riêng trong nhóm, không phải trung bình điểm tài liệu.

| Nhóm | Số tài liệu | EN→VI | VI→EN |
|---|---:|---|---|
| Bat_dong_san | 5 | 58.36 → 58.37 → 58.52 | 44.09 → 44.05 → 43.52 |
| Dich_vu_phap_ly | 5 | 71.09 → 71.07 → 71.25 | 47.13 → 47.17 → 46.79 |
| Quyen_dan_su | 12 | 49.45 → 49.42 → 49.24 | 34.13 → 34.44 → 34.03 |
| So_huu_tri_tue | 5 | 70.08 → 70.06 → 70.13 | 52.40 → 52.37 → 52.00 |
| Thu_tuc_to_tung | 4 | 70.87 → 70.85 → 70.86 | 40.90 → 41.01 → 40.38 |
| Trach_nhiem_hinh_su | 7 | 75.93 → 75.91 → 76.26 | 47.32 → 47.26 → 46.66 |
| Vi_pham_hanh_chinh | 12 | 70.12 → 70.12 → 70.19 | 45.09 → 45.08 → 44.85 |

Glossary giảm VI→EN ở cả 7 nhóm; mức giảm lớn nhất trong nhóm trách nhiệm hình sự khoảng −0,65 BLEU. EN→VI cải thiện rõ hơn ở trách nhiệm hình sự (+0,33), nhưng quyền dân sự giảm −0,21. Tập nhóm nhỏ (4–12 tài liệu), chưa đủ để kết luận chọn glossary riêng theo domain.

## Tài liệu cần xem và ví dụ thực tế

Glossary en-vi, năm tài liệu giảm mạnh nhất:

| Sample ID | Baseline | Glossary | Δ BLEU |
|---|---:|---:|---:|
| Quyen_dan_su/462 | 46.74 | 45.86 | -0.88 |
| Quyen_dan_su/139 | 56.16 | 55.42 | -0.74 |
| Quyen_dan_su/364 | 62.06 | 61.40 | -0.65 |
| Vi_pham_hanh_chinh/502 | 61.92 | 61.41 | -0.51 |
| Thu_tuc_to_tung/164 | 79.71 | 79.40 | -0.32 |

Glossary vi-en, năm tài liệu giảm mạnh nhất:

| Sample ID | Baseline | Glossary | Δ BLEU |
|---|---:|---:|---:|
| Dich_vu_phap_ly/105 | 52.11 | 48.74 | -3.36 |
| Quyen_dan_su/164 | 44.00 | 40.87 | -3.14 |
| So_huu_tri_tue/118 | 53.57 | 50.89 | -2.68 |
| Quyen_dan_su/364 | 50.99 | 48.44 | -2.55 |
| Thu_tuc_to_tung/209 | 48.17 | 45.82 | -2.36 |

Ở Quyen_dan_su/164 VI→EN, tiêu đề nguồn có “cấp thị thực, cấp thẻ tạm trú”. Baseline dịch “VISAS, TEMPORARY RESIDENCE PERMITS”; glossary đổi thành “LAISSEZ-PASSER, PERMANENT RESIDENCE CARD”, không khớp loại giấy tờ trong nguồn và reference. Trace của recovery unit u0000.r7 có gợi ý “xuất nhập cảnh translates to immigration”. Không có cặp gợi ý trực tiếp thị thực→laissez-passer ở trace này; chưa thể kết luận một cặp glossary cụ thể gây lỗi. Điểm tài liệu giảm 44,00→40,87.

Ở So_huu_tri_tue/118, nhiều chỗ đổi “Intellectual Property Law” thành “Law on Intellectual Property”. Đây là ví dụ thay cách diễn đạt có thể giảm BLEU dù nghĩa gần nhau; không phải mọi phần giảm đều là sai nghĩa. Tài liệu giảm 53,57→50,89.

Ở Dich_vu_phap_ly/105, đổi “law partnership companies”→“partnership law firms”, “Ministry of Justice”→“Justice Ministry”, cùng nhiều câu được viết lại. Điểm giảm 52,11→48,74. Cần đọc đối chiếu ngữ cảnh để tách đồng nghĩa khỏi sai nội dung.

## Recovery, cấu trúc và cảnh báo

| Bản | Chiều | Request gửi GPU | Recovery request | Retry | Unresolved unit |
|---|---|---:|---:|---:|---:|
| Baseline | en-vi | 5220 | 3609 | 465 | 7 |
| batch_length | en-vi | 5295 | 3643 | 465 | 2 |
| glossary | en-vi | 5544 | 3954 | 504 | 3 |
| Baseline | vi-en | 5295 | 3860 | 476 | 1 |
| batch_length | vi-en | 5369 | 3857 | 475 | 0 |
| glossary | vi-en | 5764 | 4423 | 551 | 0 |

Recovery là request dịch lại sau kiểm tra; không phải số tài liệu hỏng. Nó có thể trùng retry hoặc tận dụng cache, nên không cộng các cột để tính tổng request. Glossary so batching tăng recovery 8,5% EN→VI và 14,7% VI→EN. Cả hai bản mới không có OOM/lỗi vận hành, không có node rỗng unresolved, parser roundtrip 50/50 mỗi chiều.

Baseline v1.1 unresolved 7/1 unit; batching 2/0; glossary 3/0. Tuy nhiên B v1.2 replay đã đạt 2/0: không gán toàn bộ cải thiện này cho batching. Hai lỗi batching EN→VI còn ở So_huu_tri_tue/118 (protected_heading_lost) và Vi_pham_hanh_chinh/20 (line_count_mismatch 4/5). Glossary còn thêm Thu_tuc_to_tung/111 (line_count_mismatch 11/8).

| Cảnh báo (số tài liệu) | EN→VI baseline / batching / glossary | VI→EN baseline / batching / glossary |
|---|---:|---:|
| number_diff | 42 / 41 / 41 | 36 / 36 / 38 |
| number_diff_big | 19 / 19 / 19 | 4 / 4 / 6 |
| loop_not_in_source | 9 / 9 / 10 | 7 / 7 / 5 |
| heading_order | 2 / 1 / 2 | 3 / 3 / 4 |
| heading_not_as_heading | 1 / 1 / 2 | 1 / 1 / 2 |

Các cảnh báo số liệu/heading là tín hiệu tự động để rà soát, chưa phải xác nhận sai nghĩa; cách viết số/ngày và nội dung lặp có thể gây báo nhầm. Glossary VI→EN tăng number_diff_big từ 4→6 tài liệu và heading_order từ 3→4; không thể coi bản này cải thiện toàn diện dù unresolved=0.

Quyen_dan_su/390 EN→VI chỉ dài khoảng 63,5% reference ở cả ba bản. Lịch sử ABC đã ghi nhận nguồn EN khoảng 11,7k ký tự nhưng reference VI khoảng 27,5k ký tự; đây là dấu hiệu lệch nguồn/tham chiếu, không đủ căn cứ quy cho thiếu dịch. VI→EN không có tài liệu dưới 70%.

## Hiệu quả batching và tác động glossary

Padding đầu vào batching 4,53%/3,56%; glossary FIFO 29,75%/29,59%. Trung bình batch gần đủ 4 request ở cả hai bản. Batching gửi hơi nhiều request hơn baseline (5.295/5.369 so 5.220/5.295) mà vẫn nhanh hơn; tăng tốc không đến từ giảm số request. Prompt token và output token gần baseline. Điều này phù hợp với mục tiêu giảm tính toán đệm.

Baseline cũ không ghi padding, do đó không thể khẳng định mức giảm padding đối chứng. So trực tiếp batching với glossary còn trộn khác biệt prompt/recovery; không dùng chênh padding đó làm ước lượng thuần tác động batching.

Glossary so baseline tăng tổng prompt token 13,9% EN→VI và 14,1% VI→EN; generated token tăng khoảng 2,2%/3,0%. Phần tăng bao gồm gợi ý thuật ngữ và request phát sinh, không phải chỉ độ dài mỗi prompt. Có thêm chi phí nhưng chưa có lợi ích chất lượng tương xứng.

## Quyết định tiếp theo

1. Giữ batch_size=4 và dùng length batching cho bản tiếp theo; giữ glossary tắt mặc định.
2. Rà soát các cặp glossary theo ngữ cảnh, đặc biệt VI→EN và những thuật ngữ có nhiều cách dịch; đánh giá trên dev trước khi dùng public để xác nhận.
3. Nếu cần kết luận nhân quả về tốc độ, chạy thêm v1.3 FIFO/no-glossary cùng môi trường và lặp đo; đây là đề xuất, chưa chạy thêm.
4. Chưa thử kết hợp batching+glossary, nên không suy ra điểm hoặc tốc độ của bản kết hợp. Ưu tiên xử lý lỗi tiêu đề và số liệu; tách kiểm tra chất lượng cặp nguồn/tham chiếu Quyen_dan_su/390.

Dữ liệu chi tiết: reports/v13_kaggle_results.json và reports/v13_detailed_analysis.json. Script tái tạo phân tích: kaggle/public_test/analyze_v13.py.

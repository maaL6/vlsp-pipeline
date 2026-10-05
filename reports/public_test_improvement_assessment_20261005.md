# Đánh giá cải thiện SacreBLEU và chrF — 05/10/2026

Đã chấm lại 300 bản dịch A/B/C đã lưu: 50 tài liệu × 2 chiều × 3 phương án. Dùng reference nguyên bản trong CSV public test, SacreBLEU 2.6.0; không chạy thêm inference GPU. Kết quả chi tiết, theo domain/tài liệu và paired bootstrap 1.000 lần nằm trong `public_test_bleu_chrf_audit_20261005.json`.

| Phương án | EN→VI BLEU | EN→VI chrF | VI→EN BLEU | VI→EN chrF |
|---|---:|---:|---:|---:|
| A: gom + marker | 68,70 | 83,24 | 46,37 | 79,01 |
| B: gom + newline | 68,90 | 83,39 | 46,62 | 79,17 |
| C: từng node | 67,69 | 82,64 | 45,23 | 78,23 |

BLEU signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`.
chrF signature: `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`.
Đây là chrF thường; chrF++ là metric khác, được chấm thêm để chẩn đoán, không thay thế chrF.
Chưa xác minh script chấm chính thức của cuộc thi; các signature trên là cấu hình tái lập được kết quả đã lưu.

## Bằng chứng và diễn giải

- B−C: EN→VI +1,209 BLEU / +0,744 chrF; VI→EN +1,389 / +0,940. Bootstrap theo cặp tài liệu cho cả bốn delta có CI 95% lớn hơn 0. Giữ grouping làm baseline.
- B−A: EN→VI +0,202 BLEU [0,056; 0,363], +0,148 chrF [0,046; 0,265]. VI→EN +0,253 BLEU [−0,036; 0,649], +0,164 chrF [−0,033; 0,433]; chưa có bằng chứng rõ cho cải thiện chiều VI→EN. Bootstrap phản ánh biến thiên theo tài liệu, không đo biến thiên giữa các lần chạy GPU.
- B VI→EN có BLEU unigram precision 78,33%, bigram 54,57%, trigram 38,63%, 4-gram 28,60%. Có dư địa cải thiện cụm từ/cách diễn đạt; đây là suy luận chẩn đoán, không xác định duy nhất nguyên nhân.
- B EN→VI có token hyp/ref ratio 0,9434, BP 0,9418; VI→EN ratio 1,0374, BP 1. Không cố tăng độ dài đầu ra để tối ưu BP: khác phạm vi source/reference cũng ảnh hưởng ratio.
- `Quyen_dan_su/390`: nguồn EN 11.712 ký tự, nguồn VI 23.277 ký tự; VI có biểu mẫu sau phần chữ ký mà EN không có. B EN→VI BLEU 1,49/chrF 35,01; VI→EN 8,08/60,76. Không xem toàn bộ điểm thấp là lỗi bỏ dịch hoặc sinh thừa. Giữ tài liệu trong headline, thêm phân tích lệch phạm vi riêng.
- `Vi_pham_hanh_chinh/419` có ratio hyp/ref 0,480 EN→VI, 2,193 VI→EN; cần đối chiếu phạm vi và nội dung trước khi đặt recovery theo ratio.
- `Bat_dong_san/93`: payload tiêu đề VI giữ đúng “THỊ TRƯỜNG BẤT ĐỘNG SẢN”, nhưng B/C cùng dịch thành “SCHOOL’S … INDUSTRIES”. `Quyen_dan_su/454`: tiêu đề VI có “MỒ CÔI … TÀN TẬT”, B/C cùng có dịch sai nghiêm trọng. C không khắc phục được lỗi bằng cách gửi từng node. Chữ hoa là yếu tố cần ablation, chưa được chứng minh là nguyên nhân.
- Log B v1.1 có 465/2.213 và 476/2.296 unit cần recovery (~21%). Có nhiều warning số/nhãn, nhưng không được coi tất cả là lỗi thật. Đặc biệt negation/modal và lặp dòng cần kiểm precision; văn bản luật có lặp hợp lệ.
- Replay B v1.2 cải thiện unresolved, điểm gần như giữ nguyên: 68,90/83,38 và 46,61/79,17. Đây là cải thiện cấu trúc, chưa phải tăng điểm.
- Audit tiền xử lý trước đó có 0 node chia dài ở lượt đầu, 0 unit vượt cap; sửa Split và hợp nhất hàm Payload có ưu tiên thấp nếu mục tiêu chính là tăng điểm.

## Các phương án nên thử, theo thứ tự

1. **Glossary-only trên config B, ưu tiên VI→EN.** Bật `soft_glossary`, giữ `source_context=false`, `target_history=false`, newline và cap hiện tại. HY_TERMS đã thuộc format SFT của A1 theo mã nguồn. Chọn 4–8 thuật ngữ xuất hiện thật trong payload; dùng thuật ngữ xác minh từ train/glossary độc lập, không trích reference public để đưa vào inference. Thử một cấu hình glossary trước khi kết hợp với packing/casing.
2. **Ablation casing cho title/header viết hoa.** Thử chuyển payload sang cách viết hoa thông thường trước dịch; giữ loại node để phục hồi kiểu trình bày ở đầu ra khi phù hợp. Không đổi tất cả đoạn văn hay tên người. Chạy riêng và đo tổng corpus: tiêu đề chỉ chiếm một phần nhỏ văn bản, không dự báo mức tăng điểm từ vài ví dụ.
3. **Packing theo rủi ro và cấu trúc.** Giữ 768 token/8 node và newline; ưu tiên nhóm heading Điều + nội dung đầu Điều, kết thúc nhóm ở ranh giới Điều khi hợp lý; pack bảng theo hàng. Không cấm tuyệt đối ghép các Điều ngắn ngay từ đầu, vì grouping đã có lợi. Việc khoảng một phần ba unit chứa nhiều Điều chưa chứng minh rằng chúng gây giảm điểm. Chạy text-boundary và table-row thành hai ablation.
4. **Recovery/selection cho lỗi nội dung đáng tin.** Kiểm các warning trên mẫu public, phân biệt đổi định dạng số với thiếu số thật; retry có chọn lọc cho thiếu số/nhãn hoặc lặp đã xác minh. Khi hai candidate cùng hard count, đánh giá coverage và số/nhãn thay vì luôn giữ lượt đầu. Không chọn candidate bằng reference lúc inference.
5. **Context nguồn trong recovery hoặc SFT bổ sung nếu còn thời gian.** Mục tiêu giữ ngữ cảnh khi B phải dịch lại từng node. Chỉ dùng train để SFT; sample format khớp newline/heading/table. Đây là phương án chi phí cao hơn và có rủi ro thêm text context vào output.

## Thiết kế so sánh trên public test

- R0: baseline B v1.2 thực trên GPU; R1: R0 + glossary; R2: R0 + title casing; R3: R0 + packing Điều; R4: R0 + pack bảng theo hàng. Chạy từng yếu tố riêng, chỉ kết hợp các finalist tốt.
- Chấm corpus trên đủ 50 tài liệu mỗi chiều, raw reference, cùng signature; không lấy trung bình điểm tài liệu làm headline.
- Báo delta BLEU và chrF theo chiều, paired bootstrap CI theo tài liệu, missing output, unresolved, lỗi nội dung đã xác minh, recovery, runtime.
- Chưa biết quy tắc gộp hai metric của BTC: ưu tiên cấu hình tăng cả hai; nếu trade-off thì giữ các phương án không bị phương án khác vượt ở cả hai metric, chưa tự đặt trọng số tổng hợp.
- Chọn phương án trên public test theo yêu cầu hiện tại, nhưng mức tăng đo được là kết quả khám phá trên public và chưa chứng minh tăng trên private test. Giới hạn số cấu hình thử, giữ lịch sử mọi kết quả.

Nguồn định nghĩa metric: https://github.com/mjpost/sacrebleu#chrf--chrf

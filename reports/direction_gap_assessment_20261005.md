# Chênh lệch EN→VI và VI→EN — 05/10/2026

Đối chiếu 50 cặp public test, chấm lại output B v1.1 và đọc trace GPU đã lưu. Không chạy inference mới hoặc thay pipeline. Số 80/60 người dùng nhắc tới chưa xác định metric/run; không đồng nhất với kết quả local dưới đây.

| Chỉ số B v1.1 | EN→VI | VI→EN |
|---|---:|---:|
| Corpus BLEU raw, 13a | 68,90 | 46,62 |
| chrF | 83,39 | 79,17 |
| Runtime inference | 4,54 giờ | 2,87 giờ |
| Request gửi GPU | 5.220 | 5.295 |
| Token sinh, không tính cache hit | 921.193 | 589.880 |
| Unit cần recovery / unit lượt đầu | 465 / 2.213 | 476 / 2.296 |
| BP | 0,9418 | 1,0000 |
| Hyp/ref token ratio theo BLEU | 0,9434 | 1,0374 |
| Unigram precision | 93,00% | 78,33% |
| 4-gram precision | 58,06% | 28,60% |

## Tốc độ: số token đích là lời giải thích chính

EN→VI sinh nhiều token hơn 1,562 lần, runtime dài hơn 1,583 lần. Tỷ số generated_tokens/runtime gần nhau: 56,3 và 57,1 token/giây. Đây là throughput hữu ích của toàn run, không phải phép đo decode GPU riêng.

Cùng tokenizer HY, reference VI có 843.183 token so với 537.622 token EN (1,568 lần). Output cuối VI/EN có 791.486/511.344 token (1,548 lần). Bất đối xứng về số token tồn tại ngay trong dữ liệu tham chiếu, không chỉ trong output model. Backend sinh tự hồi quy và batch chờ chuỗi dài; số bước decode là lời giải thích phù hợp với log. Hai chiều chạy trên hai T4 riêng, một lần, nên không tách hoàn toàn ảnh hưởng phần cứng/batch.

Recovery gần bằng nhau (~21% unit), VI→EN còn có nhiều request hơn. Không có bằng chứng VI→EN nhanh nhờ ít retry hoặc nhờ cắt ngắn hàng loạt. Trace lượt đầu VI→EN có 9 no_eos/2.296 unit; EN→VI có 21/2.213. Bộ đếm trace có cache hit, khác token tổng của run.

## Điểm: lệch cách diễn đạt và lỗi nghĩa cùng tồn tại

Khoảng cách BLEU là 22,28 nhưng chrF là 4,22. BLEU đo mức khớp n-gram token với một reference, chrF đo n-gram ký tự; không suy diễn trực tiếp khoảng cách thành phần trăm dịch đúng/sai. 13a không thực hiện phân đoạn từ tiếng Việt: một cụm nhiều âm tiết có thể tạo nhiều token và n-gram khớp. Đây là yếu tố khiến thang điểm giữa hai ngôn ngữ đích không tương đương; chưa định lượng phần đóng góp của nó.

VI→EN không bị brevity penalty (BP=1), output dài hơn reference 3,74% theo token BLEU. Giảm mạnh precision từ unigram 78,33% xuống 4-gram 28,60% phù hợp với khác biệt cụm từ, trật tự và văn phong, nhưng không phân biệt được hoàn toàn paraphrase hợp lệ với dịch sai.

Ví dụ kiểm trực tiếp:

- `Dich_vu_phap_ly/67`: source “CHỦ TỊCH” trong ngữ cảnh Chủ tịch nước, output “CHAIRMAN”, reference “PRESIDENT”; “NAY CÔNG BỐ” thành “NAY PUBLIC DECLARATION”. Có lỗi thật và dấu hiệu thiếu ngữ cảnh giữa các dòng.
- `Bat_dong_san/93`: “THỊ TRƯỜNG BẤT ĐỘNG SẢN” trong tiêu đề thành “SCHOOL’S … INDUSTRIES”. Đây là lỗi nghĩa, không chỉ khác reference.
- `Quyen_dan_su/454`: tiêu đề về nhận con nuôi trẻ mồ côi/bị bỏ rơi/tàn tật thành nội dung “PERMANENT REINTATEMENT … ABDUCTED …”. Đoạn thân Article 1 diễn đạt cùng chủ đề tốt hơn. Chữ hoa hoặc prompt grouping là giả thuyết cần ablation, chưa phải nguyên nhân đã chứng minh.
- Cùng tài liệu 454, reference dùng “Basing upon”, output dùng “Pursuant to”; “will come into force” so với “takes effect”. Những khác biệt này góp phần giảm khớp n-gram dù có thể bảo toàn nghĩa.

BLEU lowercase đạt 70,03/48,00: khác hoa/thường ở bước chấm chỉ giải thích phần nhỏ. Không đồng nhất phép chấm lowercase với thử đổi casing nguồn trước khi dịch.

## Các giả thuyết đã kiểm tra

- Lệch source/reference ở `Quyen_dan_su/390` và nghi vấn `Vi_pham_hanh_chinh/419`: bỏ riêng hai ca để chẩn đoán được 70,93/48,22 BLEU, vẫn lệch 22,70. Không phải nguyên nhân chính của khoảng cách hai chiều. Headline vẫn giữ đủ 50 tài liệu.
- Markdown/Unicode: báo cáo scores gốc cho raw/NFC/text đều còn khoảng cách tương tự; không phải nguyên nhân chính.
- Packing/marker: A/B/C đều có khoảng cách khoảng 22 điểm. B cải thiện so với C khoảng 1,2–1,4 BLEU, không đủ giải thích chênh hai chiều.
- Chất lượng SFT, mất cân bằng train hoặc ghi nhớ dữ liệu: chưa có dữ liệu train/checkpoint provenance đầy đủ để kết luận. Cần so base với A1 và audit overlap theo tài liệu.
- Validator/selection chủ yếu bảo đảm cấu trúc; các warning nội dung không quyết định recovery và selection chỉ so số hard failures. Vì vậy trạng thái completed không bảo đảm đúng thuật ngữ hoặc nghĩa.

## Kiểm chứng tiếp theo

Ưu tiên VI→EN: glossary thuật ngữ pháp lý độc lập với reference public; ablation casing cho tiêu đề; giữ context khi dịch dòng/chức danh; so base với A1 trên dev tách theo tài liệu. Đánh giá thủ công một mẫu câu để tách paraphrase với lỗi nghĩa, báo BLEU cùng chrF và lỗi nội dung. Không tăng budget hoặc cố kéo dài bản dịch chỉ vì VI→EN điểm thấp: hiện BP=1 và ít no_eos.

Chi tiết tái chấm: `direction_gap_audit_20261005.json`. Định nghĩa metric: [SacreBLEU](https://github.com/mjpost/sacrebleu), [tokenizer 13a](https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/tokenizers/tokenizer_13a.py).

# Tạo submission VLSP 2026 trên AIHub

Script `tools/build_submission.py` đóng gói output Markdown của pipeline thành một ZIP để nộp.
Script chạy bằng thư viện chuẩn Python, không cần model, GPU, pandas hay cài thêm dependency.
Chạy các lệnh bên dưới từ thư mục gốc repository.

Hướng dẫn chính thức: [AIHub — Evaluation và Submission guidelines](https://aihub.ml/competitions/1033#learn_the_details-evaluation)
(đã đối chiếu ngày 05/10/2026).

## Đầu vào và đầu ra

- Template CSV có đúng ba cột theo thứ tự: `sample_id,direction,translation`.
- Thư mục bản dịch EN→VI và VI→EN, mỗi thư mục giữ cây `category/id.md`.
- Đầu ra là ZIP chỉ chứa một file `results.csv` tại gốc archive.

Template mặc định là `dataset/sample_submission/results.csv`: 100 bản ghi, gồm 50 ID,
mỗi ID có một dòng `en-vi` và một dòng `vi-en`. Script chỉ dùng ID, chiều dịch và thứ tự từ
template; toàn bộ bản dịch mẫu được thay bằng output của pipeline.

Ví dụ hai file ứng với cùng ID:

```text
OUTPUT_EN_VI/Bat_dong_san/137.md  → Bat_dong_san/137,en-vi
OUTPUT_VI_EN/Bat_dong_san/137.md  → Bat_dong_san/137,vi-en
```

`en-vi` là dịch nguồn tiếng Anh sang tiếng Việt; `vi-en` là dịch nguồn tiếng Việt sang tiếng Anh.
Ghép bằng cặp `(sample_id, direction)`, vì chỉ dùng ID sẽ nhầm hai chiều.

## Chạy với output của run-dir

Sau khi chạy inference hai chiều và có đủ bản dịch:

```bash
python tools/build_submission.py --en-vi OUTPUT_EN_VI --vi-en OUTPUT_VI_EN --out submission.zip
```

Thay `OUTPUT_EN_VI`, `OUTPUT_VI_EN` bằng thư mục output thật. Nếu đường dẫn chứa khoảng trắng,
đặt trong dấu nháy kép. Lệnh một dòng này dùng được trong PowerShell và Bash.

Ví dụ dùng các bản dịch public test B đã lưu trong repository:

```bash
python tools/build_submission.py --en-vi results/public_test_20261004/B_gom_node_khong_marker/predictions/en --vi-en results/public_test_20261004/B_gom_node_khong_marker/predictions/vi --out submission.zip
```

Trong run lịch sử này, thư mục `en`/`vi` được đặt theo **ngôn ngữ nguồn**:
`predictions/en` chứa bản dịch EN→VI, `predictions/vi` chứa bản dịch VI→EN.
Ví dụ này đóng gói lại output B đã lưu; để nộp output của một run mới, dùng thư mục của run đó.

## Chạy với output của docmt run

`docmt run` lưu bản dịch dưới `predictions/<direction>/`. Trỏ tới từng thư mục chiều dịch:

```bash
python tools/build_submission.py --en-vi RUN_DIR/predictions/en-vi --vi-en RUN_DIR/predictions/vi-en --out submission.zip
```

Hai chiều có thể nằm ở hai run khác nhau; khi đó truyền đường dẫn tương ứng cho mỗi chiều.

## Các tham số

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--template` | CSV xác định danh sách ID/chiều phải nộp | `dataset/sample_submission/results.csv` |
| `--en-vi` | Thư mục `.md` đã dịch sang tiếng Việt | Phải truyền nếu template có `en-vi` |
| `--vi-en` | Thư mục `.md` đã dịch sang tiếng Anh | Phải truyền nếu template có `vi-en` |
| `--out` | Đường dẫn ZIP đầu ra | `submission.zip` |

Đọc trợ giúp bằng `python tools/build_submission.py --help`.

## Cách script xử lý

1. Đọc template và kiểm tra header, ID, direction, key trùng.
2. Với từng dòng, đọc `<thư mục chiều dịch>/<sample_id>.md` bằng UTF-8.
3. Giữ nguyên nội dung Markdown, dấu nháy, dấu phẩy, khoảng trắng và xuống dòng.
4. Ghi CSV UTF-8 với mọi trường được quote; dấu nháy trong trường được nhân đôi bằng thư viện `csv`.
5. Tạo ZIP tạm rồi thay file ZIP đích khi việc đóng gói hoàn tất.

Không ghép các đoạn thành một dòng, bỏ Markdown hoặc sửa câu chữ. Script chỉ kiểm tra định dạng
submission; chất lượng, độ đầy đủ và cấu trúc bản dịch vẫn cần được kiểm tra ở output inference.
File dư trong các thư mục output không được đưa vào submission: template quyết định danh sách nộp.

CSV chứa cả tài liệu nhiều dòng trong một ô, nên số dòng văn bản vật lý không phải số bản ghi.
Script tăng giới hạn đọc ô CSV lên 100 triệu ký tự để đọc được tài liệu dài trong mẫu.

## Lỗi và cách xử lý

| Thông báo | Cách xử lý |
|---|---|
| `Missing prediction` | Kiểm tra đúng thư mục chiều dịch; chạy lại tài liệu thiếu |
| `Empty prediction` | Kiểm tra log inference và tạo lại bản dịch rỗng |
| `Duplicate template key` | Giữ đúng một dòng cho mỗi cặp ID/chiều |
| `Invalid direction` | Dùng đúng `en-vi` hoặc `vi-en` như test input |
| `Missing output directory` | Truyền thư mục cho chiều xuất hiện trong template |
| `Template columns must be` / `Malformed CSV record` | Dùng CSV chuẩn với đúng ba cột |
| `Invalid sample_id` | Dùng ID tương đối đúng theo test input, dạng `category/id` |
| `Output must have a .zip extension` | Đặt `--out` với phần mở rộng `.zip` |

Lỗi trả exit code 1 và không tạo/thay ZIP đích. Script không lấy bản dịch mẫu hoặc nguồn làm fallback.
Thành công trả exit code 0 và in số bản ghi đã đóng gói. Nếu ZIP đích đã có, chạy thành công sẽ thay nó.

## Private test

Public test yêu cầu hai chiều cho mỗi ID. Private test yêu cầu đúng chiều được cung cấp cho từng
tài liệu, tổng cộng 100 bản ghi theo hướng dẫn hiện tại. Không tái dùng danh sách public test.

Dùng template riêng có đúng ID/chiều của private test; cột `translation` trong template có thể rỗng:

```bash
python tools/build_submission.py --template PRIVATE_TEMPLATE.csv --en-vi PRIVATE_OUTPUT_EN_VI --vi-en PRIVATE_OUTPUT_VI_EN --out submission_private.zip
```

Script không cố định số bản ghi và không ép mỗi ID có hai chiều. Nếu template chỉ có một chiều,
chỉ cần truyền thư mục cho chiều đó. Người chạy chịu trách nhiệm chọn template khớp phase test.

## Nộp lên AIHub

ZIP có cấu trúc:

```text
submission.zip
└── results.csv
```

Upload ZIP này trong phase tương ứng trên AIHub. Không đặt CSV trong thư mục con, không thêm log,
model hoặc metadata vào ZIP. Các key phải khớp test set, mỗi key đúng một lần, bản dịch không rỗng.
Thứ tự dòng tùy ý theo hướng dẫn; script giữ thứ tự template để dễ kiểm tra.

AIHub chấm BLEU và chrF riêng cho từng chiều rồi macro-average khi phase có cả hai chiều.
Đóng gói submission không chạy lại inference và không tính điểm đánh giá.

## Kiểm chứng đã thực hiện

Ngày 05/10/2026, đã đóng gói thử trong thư mục tạm và đọc lại ZIP/CSV để kiểm tra:

- Output public test B: đủ 100 bản ghi, mỗi trường `translation` khớp nguyên nội dung `.md`.
- ZIP chỉ có `results.csv` ở gốc.
- Unicode, dấu phẩy, dấu nháy, xuống dòng LF/CRLF và ô dài hơn 131.072 ký tự đọc lại đúng.
- Template một chiều hoạt động mà không yêu cầu thư mục chiều còn lại.
- Key trùng, direction sai, file thiếu, bản dịch rỗng, thiếu thư mục chiều và template không có
  bản ghi đều bị từ chối; ZIP đích có sẵn được giữ nguyên khi lỗi.

Đây là kiểm tra local; chưa upload hay chạy bộ chấm của AIHub.

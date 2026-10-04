# CHANGE_NOTES — pipeline-v1

## v1.2 (04/10/2026) — B: unit gom bị lỗi luôn được dịch lại từng node

**Vấn đề** (log run B, v1.1): 7 unit en-vi + 1 vi-en còn lỗi sau recovery. Phần lớn là lệch dòng mà số dòng
vẫn khớp (model tách một câu thành 2 dòng rồi gộp tiêu đề với đoạn sau) → chỉ lộ ra qua
`protected_heading_lost`. Lỗi này không thuộc `MARKER_CODES` nên s08 dùng *regroup* (nhóm ≤256 token,
vẫn nối xuống dòng); model lại gộp tiêu đề → `line_count_mismatch`, recovery tệ hơn → giữ lượt đầu bị lệch.
Ví dụ `Vi_pham_hanh_chinh/289`: dòng "c) …" bịa thêm, "5./6." lệch một ô, "Điều 15." dính vào "Phạt tiền…".

**Thay đổi**: `s08_recovery.recovery_mode`: unit nhiều node nối xuống dòng (`Unit.markers == []`) bị lỗi hard
bất kỳ → luôn *independent* (từng node). Unit có marker (A) và unit 1 node giữ nguyên quy tắc cũ.
Chỉ áp dụng ở recovery; lượt đầu, kiểm tra, chọn bản giữ nguyên.

**Đo (không GPU)**: chạy v1.2 trên public test với ReplayBackend lấy output từ trace GPU của run B và run C
(request "từng node" của recovery mới giống hệt request lượt đầu của C): 0 request thiếu.
`reports/public_test_b_v12_replay_20261004.json`:

| | B v1.1 | B v1.2 |
|---|---|---|
| BLEU en-vi / vi-en | 68.90 / 46.62 | 68.90 / 46.61 |
| Doc `completed` en-vi / vi-en | 44 / 49 | **48 / 50** |
| Unit unresolved en-vi / vi-en | 7 / 1 | 2 / 0 |
| Request en-vi / vi-en | 5.822 / 6.156 | 5.856 / 6.161 |

BLEU không đổi (5 file đổi, mỗi doc ±0,13); lợi ích là cấu trúc (4 doc hết lệch tiêu đề/đoạn). Còn lại:
`So_huu_tri_tue/118` (dịch lại từng node không ít lỗi hơn, giữ lượt đầu) và `Vi_pham_hanh_chinh/20`.
Lưu ý: output của C sinh trong batch fp16 khác → run GPU thật có thể lệch vài token.

**Nghiệm thu**: `tools/verify.sh` 61/61 test, replay `20261003b` + mock 4 nhánh (config A) giống từng byte;
code commit tái tạo đúng 100/100 bản dịch của mô phỏng.

## v1.1 (04/10/2026) — phương án B: gom node, không marker (mặc định mới)

**Quyết định** (Lâm, 04/10/2026): chọn B sau khi so 3 phương án trên full public test (50 doc × en-vi/vi-en,
A1 merged, HF fp16 greedy rep 1.05, batch 4, Kaggle T4; `reports/public_test_abc_20261004.json`):

| Phương án | Config | BLEU en-vi | BLEU vi-en | Thời gian en-vi / vi-en |
|---|---|---|---|---|
| A. Gom node + marker `[SEG_…]` | `hy_a1_structure.json` | 68.70 | 46.37 | 5,5 h / 3,1 h |
| **B. Gom node + không marker** | `hy_a1_structure_newline.json` | **68.90** | **46.62** | 4,5 h / 2,9 h |
| C. Không gom node + không marker | `hy_a1_structure_nomarker.json` | 67.69 | 45.23 | 4,3 h / 2,9 h |

Bootstrap 50 doc (khoảng 95%): B − A = +0,20 [+0,05, +0,37] en-vi, +0,25 [−0,05, +0,65] vi-en;
A − C = +1,01 / +1,14 và B − C = +1,21 / +1,39 (đều > 0). Lượt đầu: A mất marker ở 771/2.214 unit en-vi
(35%), B lệch số dòng ở 465/2.213 en-vi và 476/2.296 vi-en (21%); C gần như không lỗi nhưng không có ngữ cảnh.

**Thay đổi code** (chỉ có hiệu lực khi `extra.payload_delimiter = "newline"`; mặc định `"marker"` giữ nguyên A):
- `s03_pack`: `payload_delimiter(cfg)`, `use_markers(cfg)`, `unit_markers(ns, indices, cfg)` (thay 2 biểu thức
  `marks = …` lặp lại); `render_payload` với unit nhiều node và `markers == []` → `"\n".join(payload)`.
  Chú thích đầy đủ ở phần *payload delimiter* trong file.
- `s06_parse_output`: unit nhiều node không có marker → `_parse_lines`: dòng không rỗng thứ i → node i;
  số dòng khác số node → hard `line_count_mismatch:<có>/<cần>`, không gán gì.
- `s08_recovery`: `line_count_mismatch` thuộc `MARKER_CODES` → unit được dịch lại từng node như lỗi marker;
  regroup dùng `unit_markers` nên nhóm dịch lại cũng nối bằng xuống dòng.
- `run.sh`: `CONFIG` mặc định → `configs/hy_a1_structure_newline.json`.
- Mới: `configs/hy_a1_structure_newline.json` (B), `configs/hy_a1_structure_nomarker.json` (C),
  `tests/test_newline_delimiter.py` (5 test), `kaggle/public_test/` (script Kaggle + chấm),
  `reports/public_test_abc_20261004.json`.

**Nghiệm thu**: `tools/verify.sh` 60/60 test; replay trace `20261003b` và mock 4 nhánh với config A giống từng
byte như trước (A không đổi). Replay trace Kaggle của run B qua code commit này: 50/50 en-vi + 50/50 vi-en
bản dịch giống từng byte, 0 request thiếu trong trace.

**Còn mở**: lệch số dòng 21% unit làm mất ngữ cảnh khi dịch lại từng node (thử dịch lại theo nhóm 2–3 node);
en-vi B còn 7 unit unresolved (A: 2); `run.sh` với model thật chưa kiểm (kernel gọi `run-dir` trực tiếp).
Mục 3 bên dưới (mất marker ở s03 pack) không còn áp dụng cho B.

## v1.0 (03/10/2026) — tách module, hành vi không đổi

Nguồn: `report_lam_3_10/code/lam_docmt` (bản sửa recovery cap; hash `units.py` 9d7df1be…, `runtime.py`
65f2ce2f… như runner `structure_fix`). Thư mục gốc chỉ được đọc.

**Tách cơ học** (`tools/split_from_lam_docmt.py`): mọi hàm/lớp/hằng chép nguyên văn sang đúng một file
mới, import tự sinh; sửa đường dẫn sau tách ghi trong `POST_EDITS` của script. `text_normalize.py` chép
nguyên byte.

**Chuyển logic khỏi `_DocJob`** sang file bước, thân hàm giữ nguyên:
s04 `render_unit_prompt`; s05 `fits_alone`, `recovery_source_tokens`, `fit_prompt`; s06 `leak_pattern`,
`strip_internal_markers`; s07 `combine_hard`; s08 `should_recover`, `recovery_mode`, `regroup_is_noop`,
`recovery_reason`, `is_identical_retry`; s09 `first_candidate`, `recovery_candidate`, `choose_candidate`,
`fill_nodes`, `is_accepted`; s10 `assemble`, `strip_marker_leaks`, `decide_status`.

**Khác lam_docmt (không đổi output dịch)**:
- Backend llama.cpp bỏ; `RunConfig.backend` mặc định `"hf"` (trước `"llamacpp"`) → `config_sha256` khác
  các run cũ dù cùng tham số. `server_url` giữ lại (không dùng) để đọc được run_config cũ.
- Đổi tên để không che module chuẩn Python: `types.py` → `core/datatypes.py`, `trace` → `tracing/`,
  không dùng tên `io`.
- `code_sha256` trong run.json tính trên `docmt/**` (khác giá trị cũ).
- Mới: `run-dir` (folder → folder, log ở `<out>.work`), `inspect --stop-after`, `--config` preset JSON,
  `backends/replay.py`, `run.sh`, công cụ nghiệm thu. Tool/train port ghi kết quả vào `pipeline-v1/`.
- Không còn đường dẫn tuyệt đối: dữ liệu test lấy từ `<repo>/task 3-10/CHUNG/inputs` hoặc
  `DOCMT_TEST_INPUTS`; thiếu thì pytest dừng với thông báo. `run.sh` nhận `BACKEND=mock` để kiểm cài đặt.

**Nghiệm thu**: xem README › Nghiệm thu (55/55 test; replay + mock 4 nhánh giống từng byte; parse 84/84).

## Phát hiện khi tách (chưa sửa — để các bước sau, mỗi mục một thay đổi)

1. Hai bản `unwrap`/payload: `s02_payload.make_payload` (parse) và `s03_split.payload_of` (mảnh tách).
   Đã kiểm 19.257 node translatable của 84 doc-direction: kết quả giống nhau 100% → gộp được, rủi ro thấp.
2. s07: `loop_not_in_source`, `length_ratio_low`, `fffd` chỉ là warning → không kích hoạt recovery
   (ca chép ô "mitochondrial" ở 41 en-vi, run `b`).
3. s03 pack: 335 node mất marker ở lượt đầu run `b` (27% unit vào recovery); ô bảng/tiêu đề ngắn gom chung
   unit có marker là bối cảnh của lỗi chép ô bên cạnh.
4. Chưa có kiểm từ chức danh ký (DEPUTY ↔ PHÓ, KT., TM., PP.): ca "DEPUTY PRIME MINISTER" → "THỦ TƯỚNG" ở 111.
5. `run-dir` khi một tài liệu `failed` không ghi file output (exit code ≠ 0, liệt kê `missing`). Chính sách
   fallback cho bản nộp (chép nguồn? dịch lại?) cần thống nhất trước private test.

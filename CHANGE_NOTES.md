# CHANGE_NOTES — pipeline-v1

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

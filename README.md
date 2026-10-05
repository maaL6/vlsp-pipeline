# pipeline-v1 — HY A1 document MT, one module per step

Pipeline dịch tài liệu Markdown EN↔VI cho VLSP 2026, tách từ `report_lam_3_10/code/lam_docmt` (bản đã sửa
recovery cap, run `20261003c`) thành **một file cho mỗi bước**, để sửa/tối ưu từng bước độc lập.

**v1 = refactor thuần**: hành vi giống `lam_docmt` từng byte (xem *Nghiệm thu*). Các sửa chất lượng
(lặp / thiếu / EOS sớm) làm ở các bước sau, mỗi sửa một thay đổi, ghi trong `CHANGE_NOTES.md`.
`report_lam_3_10/` không bị sửa.

**Phương án chốt (04/10/2026): B = gom node + không marker** (`configs/hy_a1_structure_newline.json`,
mặc định của `run.sh`). Các node của một unit được gom như trước (≤8 node / ≤768 token, bảng ≤16 ô)
nhưng chỉ nối bằng xuống dòng, không chèn `[SEG_…]`; output tách lại theo dòng. So trên full public test
(50 doc × 2 chiều), xem *Kết quả public test* bên dưới.

## Kết quả public test (04/10/2026)

Full pipeline, A1 merged, HF fp16 greedy, batch 4, Kaggle T4 (mỗi chiều một GPU), chấm
`benchmark/eval_markdown.py` (SacreBLEU raw, `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`).
Mỗi phương án chạy một lần; đủ 50/50 bản dịch mỗi chiều, 0 doc `failed`.

| Phương án | Config | BLEU en-vi | BLEU vi-en | Thời gian (en-vi / vi-en) |
|---|---|---|---|---|
| A. Gom node + marker | `hy_a1_structure.json` | 68.70 | 46.37 | 5,5 h / 3,1 h |
| **B. Gom node + không marker** | `hy_a1_structure_newline.json` | **68.90** | **46.62** | 4,5 h / 2,9 h |
| C. Không gom node + không marker | `hy_a1_structure_nomarker.json` | 67.69 | 45.23 | 4,3 h / 2,9 h |

- Gom node là yếu tố chính: A/B hơn C khoảng +1,0…+1,4 BLEU (bootstrap 50 doc, khoảng 95% không chứa 0).
- Bỏ marker không làm kém đi: B − A = +0,20 [+0,05, +0,37] en-vi, +0,25 [−0,05, +0,65] vi-en.
- B nhanh hơn A ~1 h ở en-vi vì ít phải dịch lại: lượt đầu A mất marker ở 35% unit en-vi, B lệch số
  dòng ở 21% unit; unit lỗi được dịch lại từng node.
- v1.2 (sau khi đo): unit gom của B bị lỗi bất kỳ được dịch lại từng node → BLEU giữ nguyên
  (68.90 / 46.61), số doc còn lỗi cấu trúc giảm 7 → 2 (đo bằng replay output GPU của B và C,
  `reports/public_test_b_v12_replay_20261004.json`; chưa chạy lại GPU).
- Output gốc của cả 3 lần chạy (bản dịch .md, điểm, log, trace để replay): `results/public_test_20261004/`.
- Chi tiết (điểm từng doc, recovery, token, VRAM): `reports/public_test_abc_20261004.json`. Cách chạy lại:
  `kaggle/public_test/README.md`. Code B được kiểm bằng replay trace GPU của chính run B: 100/100 bản dịch
  giống từng byte.

## Trạng thái (04/10/2026)

| Hạng mục | Trạng thái |
|---|---|
| Tách module, hành vi giống lam_docmt | completed — replay/mock/test, không GPU (xem *Nghiệm thu*) |
| `run-dir` folder → folder trên GPU (HF + A1) | completed — full public test 50 doc × 2 chiều, 3 phương án A/B/C trên Kaggle T4 (04/10) |
| Chọn cách gửi unit | completed — **B (gom node + không marker)**, mặc định của `run.sh` |
| `run.sh` với model thật | not_checked (kernel Kaggle gọi `python -m docmt run-dir` trực tiếp; `run.sh` mới kiểm với `BACKEND=mock`) |
| Windows | not_checked (code không phụ thuộc Linux; `run.sh` cần bash → dùng lệnh `python -m docmt run-dir` bên dưới) |
| Dockerfile | not_started |
| Sửa lặp / thiếu / EOS sớm | not_started (danh sách trong `CHANGE_NOTES.md`) |

## Giao cho người khác chạy

Folder này **không chứa model**. Người nhận cần:

1. **Model A1 merged** (~3,6 GB): `Hy-MT2-1.8B-release/models/merged/` (có `model.safetensors`
   sha256 `5dc305ae…`, `tokenizer.json` sha256 `b475bbef…`, `config.json`, `chat_template.jinja`). Đặt vào
   `pipeline-v1/models/hy_a1_merged/` hoặc trỏ biến `MODEL`.
2. **GPU NVIDIA** (~5 GB VRAM; T4 đã dùng cho các run trước, fp16). Backend `hf` gọi `.to("cuda")`, không chạy CPU.
3. **Python ≥ 3.10** (đã kiểm trên 3.12), torch bản CUDA, rồi `pip install -r requirements.txt`.
   Cần `tokenizers ≥ 0.21` và `transformers ≥ 5.6`: bản cũ hơn báo lỗi
   `data did not match any variant of untagged enum ModelWrapper` khi đọc `tokenizer.json` của A1.

Kiểm cài đặt không cần GPU (mock chỉ chép lại nguồn, **không phải bản dịch**):

```bash
BACKEND=mock MODEL=/path/hy_a1_merged ./run.sh INPUT_DIR /tmp/check_out en-vi
```

Chạy thật:

```bash
MODEL=/path/hy_a1_merged ./run.sh INPUT_DIR OUTPUT_DIR en-vi      # Linux / macOS / Git Bash
# Windows (không có bash), chạy trong thư mục pipeline-v1:
python -m docmt run-dir --input-dir IN --output-dir OUT --direction en-vi ^
    --config configs/hy_a1_structure_newline.json --run-id RUN_ID --backend hf ^
    --model C:\path\hy_a1_merged --tokenizer C:\path\hy_a1_merged\tokenizer.json
```

`OUTPUT_DIR` chỉ chứa các file `.md` dịch (cùng cây thư mục với `INPUT_DIR`). Log, trace, cache ở
`OUTPUT_DIR.work/` (`logs/document_status.jsonl` cho trạng thái từng tài liệu). Exit code ≠ 0 khi có tài
liệu `failed` hoặc thiếu output; khi đó **không có file output** cho tài liệu đó (chính sách fallback chưa
chốt). Chạy lại vào cùng `WORK_DIR` thì dùng cache resume (request đã có output không gửi lại); đổi
config thì phải dùng `WORK_DIR` mới (runtime từ chối ghi run khác config vào thư mục cũ).

Chỉ dùng trong repo VLSP26 (cần thư mục anh em, không đi kèm folder): `tests/` (cần `task 3-10/CHUNG/inputs`,
hoặc `DOCMT_TEST_INPUTS`), `tools/verify.sh`, `tools/replay_equivalence.py`, `tools/mock_equivalence.py`
(cần `report_lam_3_10/`), `train/` (cần `report_lam_3_10/01_prompt_contract_for_Linh/token_audit.py`).
Repo git riêng (`github.com/maaL6/vlsp-pipeline`, private); clone vào `VLSP26/pipeline-v1` thì các đường dẫn
mặc định đúng, clone chỗ khác thì đặt `VLSP26_ROOT=/path/VLSP26` cho `tools/verify.sh`.

## Tạo submission AIHub

Hướng dẫn đầy đủ, ví dụ với output có sẵn, private test và xử lý lỗi: [docs/submission.md](docs/submission.md).

Sau khi dịch xong hai chiều, đóng gói các file `.md` thành ZIP:

```bash
python tools/build_submission.py --en-vi OUTPUT_EN_VI --vi-en OUTPUT_VI_EN --out submission.zip
```

Mỗi thư mục output giữ cây `category/id.md` như `run-dir`. Nếu dùng `docmt run`, trỏ vào
`RUN_DIR/predictions/en-vi` và `RUN_DIR/predictions/vi-en` tương ứng.
Script mặc định đọc `dataset/sample_submission/results.csv` để lấy danh sách `(sample_id, direction)`
và thứ tự; thay toàn bộ cột `translation` bằng nội dung file output, giữ nguyên Markdown và xuống dòng.
Không dùng bản dịch trong mẫu làm fallback. Thiếu file, bản dịch rỗng, key trùng hoặc direction sai
thì báo lỗi và không tạo/thay ZIP. File ZIP chỉ chứa `results.csv` ở gốc, đúng ba cột, UTF-8,
quote/escape bằng thư viện CSV chuẩn. Upload `submission.zip` lên AIHub.

Với private test, dùng `--template PATH_TO_PRIVATE_TEMPLATE.csv` có đúng các key của private test;
script lấy số dòng và chiều dịch từ template, không cố định 100 dòng hay bắt buộc mỗi ID có hai chiều.
Chỉ cần truyền thư mục của các chiều xuất hiện trong template.

## Chạy (chi tiết)

```bash
# Final / Docker: folder .md -> folder .md (giữ nguyên cây thư mục); log ở OUTPUT_DIR.work/
MODEL=/path/hy_a1_merged ./run.sh INPUT_DIR OUTPUT_DIR en-vi

# Dev / smoke: manifest -> delivery layout (predictions/, logs/, tests/) như lam_docmt run
python -m docmt run --config configs/hy_a1_structure_newline.json --run-id RUN_ID \
    --manifest MANIFEST --data-root ROOT --out delivery/RUN_ID \
    --model A1_MERGED --tokenizer TOKENIZER_JSON [--directions en-vi,vi-en] [--dry-run]

# Xem kết quả trung gian của một bước, không cần model/GPU
python -m docmt inspect --stop-after parse|payload|split|pack|prompt \
    --manifest MANIFEST --data-root ROOT --tokenizer TOKENIZER_JSON --out stage.jsonl

python -m docmt parse-audit       --manifest M --data-root R --tokenizer T --out parser.json --both-directions
python -m docmt validate-existing --manifest M --data-root R --pred-root PRED --out diag.json
```

Backend: `hf` (transformers, GPU CUDA) và `mock` (test). llama.cpp đã bỏ. `--set key=value` ghi đè preset.

Preset `configs/*.json`:

| File | Phương án | Khác `hy_a1_structure.json` |
|---|---|---|
| `hy_a1_structure_newline.json` | **B, mặc định `run.sh`** | `extra.payload_delimiter = "newline"` |
| `hy_a1_structure.json` | A (gom + marker) | — (= config run GPU `20261003b` trừ run_id/đường dẫn/hash; mốc của `tools/verify.sh`) |
| `hy_a1_structure_nomarker.json` | C (từng node) | `max_nodes = 1`, `max_table_nodes = 1` |
| `hy_a1_source_context*.json` | nhánh context (*tắt*) | bật source context / glossary / history |

`extra.payload_delimiter` (`"marker"` mặc định | `"newline"`) chỉ đổi cách nối node trong prompt và cách tách
output; chia đoạn, cap token, kiểm tra, recovery giữ nguyên. Giải thích đầy đủ ở đầu phần
*payload delimiter* trong `docmt/steps/s03_pack.py`.

## Các bước và file

Luồng một tài liệu (chỉ đọc nguồn, không bao giờ đọc reference):

| Bước | File | Vào → Ra | Ghi chú |
|---|---|---|---|
| s01 parse | `steps/s01_parse.py`, `steps/s01_parse_lines.py` (luật dòng/nhãn/tên), `steps/s01_parse_audit.py` | raw text → `Node[]` | `"".join(n.raw) == src`; header/bảng/chữ ký/scope |
| s02 payload | `steps/s02_payload.py` | `content_raw` → `payload` | `normalize(unwrap(raw))` |
| s03 split | `steps/s03_split.py` | `Node[]` → work nodes (`nX.1`…) | cắt theo câu → phẩy → khoảng trắng, không cắt số |
| s03 pack | `steps/s03_pack.py` | work nodes → `Unit[]` (B: nối `\n`; A: marker `[SEG_…]`) | text ≤768/8 node, bảng ≤384/16, chữ ký ≤128 |
| s04 prompt | `steps/s04_prompt.py` | unit (+evidence) → prompt | `HY_DEFAULT` / `HY_TERMS` / `HY_BACKGROUND` |
| s05 budget | `steps/s05_budget.py` | prompt → `max_new_tokens`, bỏ evidence nếu thiếu chỗ | `min(4096−prompt−32, max(256, ⌈2.5·src⌉+128))` |
| (generate) | `backends/hf.py`, `backends/mock.py`, `backends/replay.py` | `GenRequest[]` → `GenResult[]` | greedy, fp16, rep 1.05 |
| s06 parse output | `steps/s06_parse_output.py` | output → segment theo node | B: dòng i → node i, lệch số dòng = `line_count_mismatch`; A: lỗi marker, rò marker |
| s07 validate | `steps/s07_validate_unit.py` + `checks/{numbers,headings,loops,lexical}.py` | segment → hard / warnings | **hard** kích hoạt recovery; warning chỉ ghi |
| s08 recovery | `steps/s08_recovery.py` | unit lỗi → unit con | B: mọi lỗi → từng node; A: marker lỗi → từng node, khác → gom ≤256 token; 1 vòng |
| s09 select | `steps/s09_select.py` | lượt đầu vs recovery → text cuối mỗi node | ít hard hơn thắng; hoà giữ lượt đầu |
| s10 assemble | `steps/s10_assemble.py` | text node → Markdown | `lead + bản dịch + trail`, giữ layout |
| s10 doc check | `steps/s10_doc_check.py` | tài liệu → status | coverage, heading, số, lặp, U+FFFD |

Phần còn lại:

- `engine/doc_job.py` — điều phối một tài liệu: gọi các bước theo thứ tự, log lỗi, ghi trace. Không chứa
  quy tắc nghiệp vụ; muốn đổi quy tắc thì sửa file bước tương ứng.
- `engine/run.py` (một hướng của run, cache, run.json), `engine/scheduler.py` (gom request thành batch),
  `engine/api.py` (`run_manifest`, `translate_document`, layout delivery), `engine/inspect.py`.
- `context/` — evidence cho prompt: `source_index.py` (tiêu đề/định nghĩa/Điều được viện dẫn),
  `glossary.py`, `memory.py` (bản dịch đã chấp nhận), `evidence.py` (chọn + cắt ≤512 token). **Tắt** ở
  `hy_a1_structure`; chỉ bật khi có checkpoint context-SFT.
- `core/` — `datatypes.py` (Node/Unit/GenRequest/…: interface giữa các bước, giữ ổn định), `config.py`,
  `tok.py`, `text_normalize.py` (bản copy **byte-identical** của chuẩn nhóm, test kiểm), `hashing.py`.
- `inputs/sources.py` — manifest hoặc folder → `DocItem`. `tracing/` — trace, status, cache resume, smoke table.
- `docmt/train/` + `train/` — encode SFT, train context-SFT, build/audit derivative (ghi vào `train/data/`).

Thay một bước bằng bản thử nghiệm mà không sửa file gốc: `run_manifest(..., components={"check_unit": my_fn})`
(xem `Components` trong `engine/doc_job.py`).

## Đo từng bước

| Bước sửa | Đo không cần GPU | Cách |
|---|---|---|
| s01–s03 | ✅ | `parse-audit` (84/84), `inspect --stop-after parse/split/pack` rồi so JSONL |
| s04–s05 | một phần | `inspect --stop-after prompt` (token, budget); chất lượng cần GPU |
| s07, s09, s10 | ✅ | replay trace đã có (`backends/replay.py`, `tools/replay_equivalence.py`): thấy ngay output/status đổi thế nào |
| s08, decoding | ❌ | prompt recovery mới không có trong trace → chạy GPU (smoke 9 tài liệu) |

## Nghiệm thu refactor (`tools/verify.sh PYTHON`)

Kết quả hiện tại (03/10/2026, không GPU):

- `pytest`: **61/61** (50 test port từ lam_docmt + 5 test v1 + 6 test phương án B, 04/10).
- Replay/mock bên dưới chạy với config A (`hy_a1_structure.json`): chứng minh thêm phương án B không đổi
  hành vi của A. Hành vi của B được kiểm riêng bằng replay trace Kaggle của run B (100/100 giống từng byte).
- Replay trace GPU run `20261003b` (smoke6), lam_docmt vs v1: prediction **6/6**, document_status **6/6**,
  trace **1111/1111**, prompt **972/972** giống từng byte → `reports/replay_equivalence.json`.
  2 request không có trong trace (`111:u0334.r4/.r5`) = 2 mảnh mới của bản sửa recovery cap (369 → 233+136);
  vì vậy 5/6 prediction trùng run `b`, `111` khác — đúng dự kiến, giống nhau giữa lam_docmt và v1.
- Mock 4 nhánh (structure, source_context, +glossary, +history), lam_docmt vs v1: giống từng byte →
  `reports/mock_equivalence.json` (có HY_BACKGROUND/HY_TERMS, evidence title/definition/reference/glossary/history).
- `parse-audit` 84/84 roundtrip; totals và từng tài liệu giống báo cáo đã bàn giao (chỉ khác thời gian) →
  `reports/parser_roundtrip_84.json`. `tools/audit_recovery_caps.py` giống hệt báo cáo cũ.

Mọi thay đổi **không** được phép đổi hành vi phải qua `tools/verify.sh`. Thay đổi **có chủ đích** đổi
hành vi thì replay sẽ báo khác: đó là bằng chứng cần ghi vào `CHANGE_NOTES.md`.

## Cấu trúc thư mục và vai trò

Ký hiệu cột **Chạy dịch**: **cần** = thiếu thì `run.sh` không chạy; *dev* = chỉ để phát triển/kiểm tra;
*tắt* = có code nhưng không dùng ở nhánh chính (B, `hy_a1_structure_newline.json`).

```
pipeline-v1/
├── run.sh  requirements.txt  README.md  CHANGE_NOTES.md  .gitignore
├── configs/   docmt/   tests/   tools/   train/   reports/
```

### Cấp gốc

| Thư mục / file | Tác dụng | Chạy dịch |
|---|---|---|
| `run.sh` | Lệnh cuối: folder `.md` → folder `.md` (gọi `python -m docmt run-dir`) | **cần** |
| `requirements.txt` | Dependency (torch CUDA cài riêng) | **cần** |
| `configs/` | Preset JSON mỗi nhánh (bảng ở *Chạy (chi tiết)*). `hy_a1_structure_newline.json` (B) là nhánh chính, `run.sh` đọc mặc định | **cần** (`hy_a1_structure_newline.json`); A, C để so sánh/nghiệm thu; 3 preset context *tắt* |
| `docmt/` | Toàn bộ code pipeline (bảng bên dưới) | **cần** |
| `tests/` | 61 test pytest; cần `task 3-10/CHUNG/inputs` (hoặc `DOCMT_TEST_INPUTS`) | *dev* |
| `tools/` | Nghiệm thu và audit (bảng bên dưới) | *dev* |
| `train/` | `build_hy_derivative.py`, `audit_hy_derivative.py`: tạo/kiểm dữ liệu context-SFT; đọc `report_lam_3_10/.../token_audit.py`, ghi vào `train/data/` (bị ignore) | *dev* (train, không phải inference) |
| `reports/` | Bằng chứng nghiệm thu dạng JSON (replay, mock, parse-audit 84, recovery cap; `tools/verify.sh` tạo lại được) và `public_test_abc_20261004.json` (kết quả A/B/C); có đường dẫn máy local | không |
| `kaggle/public_test/` | Chạy full public test trên Kaggle (`push.sh`), tải kết quả và chấm (`fetch_and_compare.sh`); xem README trong thư mục | *dev* (benchmark) |
| `README.md`, `CHANGE_NOTES.md` | Hướng dẫn; lịch sử thay đổi hành vi và danh sách lỗi đang chờ sửa | không |
| `.gitignore` | Chặn `models/`, `*.safetensors`, `*.work/`, `delivery/`, `train/data/`, cache Python | không |

### Bên trong `docmt/`

| Package / file | Tác dụng | Chạy dịch |
|---|---|---|
| `cli.py`, `__main__.py` | Lệnh `run`, `run-dir`, `inspect`, `parse-audit`, `validate-existing`, `smoke-table` | **cần** |
| `core/` | `datatypes.py` (Node, Unit, GenRequest, GenResult…: interface giữa các bước), `config.py` (RunConfig, preset nhánh), `tok.py` (đếm token HY), `text_normalize.py` (copy byte-identical chuẩn nhóm), `hashing.py` (sha256, model/code hash), `codes.py` (helper nhỏ) | **cần** |
| `inputs/` | `sources.py`: manifest hoặc folder → `DocItem`, chỉ đọc phía nguồn | **cần** (1 file, có thể gộp vào `engine/`) |
| `steps/` | 10 bước của pipeline (bảng *Các bước và file*) | **cần**; riêng `s01_parse_audit.py` *dev* (lệnh `parse-audit`) |
| `checks/` | Hàm kiểm nguồn vs output dùng ở s07 và s10: `numbers.py`, `headings.py`, `loops.py`, `lexical.py` (phủ định, modal, chưa dịch), `common.py` | **cần** |
| `backends/` | `hf.py` (transformers, GPU CUDA, fp16, batch), `base.py` (interface, đếm OOM/retry), `mock.py` (echo + lỗi giả lập cho test), `replay.py` (trả output đã ghi trong trace, đo offline) | **cần** (`hf`); `mock`/`replay` *dev* |
| `engine/` | `doc_job.py` (điều phối một tài liệu: gọi bước, log, trace), `scheduler.py` (gom request thành batch), `run.py` (một hướng của run, cache, run.json), `api.py` (`run_manifest`, `translate_document`, layout delivery), `inspect.py` (dump kết quả trung gian) | **cần**; `inspect.py` *dev* |
| `tracing/` | `writer.py` (trace, status, prompts, lỗi), `cache.py` (resume theo prompt), `records.py` (bản ghi trace, environment), `smoke_table.py` (`smoke_result.csv` theo mẫu bàn giao) | **cần** (resume + log); `smoke_table.py` chỉ cho báo cáo |
| `context/` | Evidence cho prompt `HY_BACKGROUND`: `source_index.py` (tiêu đề/đường dẫn, định nghĩa, Điều được viện dẫn), `glossary.py`, `memory.py` (bản dịch đã chấp nhận), `evidence.py` (chọn + cắt ≤512 token), `common.py` | *tắt* — chỉ bật khi có checkpoint context-SFT |
| `train/` | `sft.py` (encode mẫu SFT + smoke forward/backward), `train_sft.py` (train LoRA context-SFT) | không (train) |

### `tools/`

| File | Tác dụng | Ghi chú |
|---|---|---|
| `verify.sh` | Một lệnh nghiệm thu: pytest + check tên + replay + mock 4 nhánh | Dùng sau mọi thay đổi |
| `replay_equivalence.py` | Chạy lam_docmt và v1 trên trace GPU run `20261003b`, so từng byte | Mốc so sánh hiện là lam_docmt (xem lưu ý dưới) |
| `mock_equivalence.py` | Như trên với MockBackend cho 4 nhánh (kể cả context) | như trên |
| `compare_runs.py` | BLEU/chrF++ smoke6 giữa các hệ thống (cần sacrebleu và reference) | Chỉ chẩn đoán, không dùng để tune |
| `check_names.py` | Kiểm tên global bị mất khi tách file | Chỉ hữu ích lúc tách; xoá được |
| `audit_recovery_caps.py` | Kiểm hồi quy lỗi recovery 369 > 256 token | Đã có test fixture tương ứng; xoá được |
| `split_from_lam_docmt.py` | Script tách lam_docmt thành v1 (lưu nguồn gốc) | Dùng một lần; xoá được (git history giữ) |

### Thừa hoặc cần dọn (chưa làm)

- Xoá được: `tools/split_from_lam_docmt.py`, `tools/check_names.py`, `tools/audit_recovery_caps.py`.
- Nên tách khỏi pipeline inference: `docmt/train/` và `train/` (phụ thuộc `report_lam_3_10/`).
- Tuỳ quyết định: `context/` + 3 preset context (bỏ nếu không làm source context trước freeze 08/10);
  `reports/` có thể đưa vào `.gitignore`.
- Trong code: hai hàm payload trùng nhau (`s02_payload.make_payload`, `s03_split.payload_of`, giống nhau
  trên 19.257 node); trường config `server_url` không còn dùng.
- **Lưu ý khi bắt đầu sửa chất lượng:** `replay_equivalence.py`/`mock_equivalence.py` so v1 với lam_docmt.
  Khi v1 cố ý đổi hành vi, hai công cụ này sẽ báo khác (đúng thiết kế); khi đó đổi mốc so sánh sang
  commit v1 trước đó.

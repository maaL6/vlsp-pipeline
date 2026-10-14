# vlsp-pipeline — dịch tài liệu Markdown EN↔VI (VLSP 2026)

Pipeline dịch tài liệu pháp lý Markdown EN→VI và VI→EN bằng Hy-MT2-1.8B + LoRA adapter (đã merge).
Mỗi bước xử lý nằm trong một file riêng (`docmt/steps/`).

Điểm public test (100 tài liệu, chấm raw bằng `tools/score_public.py`):

| Chiều | BLEU | chrF |
|---|---|---|
| EN→VI | 70.87 | 84.48 |
| VI→EN | 47.44 | 79.50 |

## Cài đặt

Repo **không chứa model**. Cần:

1. **GPU NVIDIA** (~5 GB VRAM, fp16, đã chạy trên T4). Backend `hf` gọi `.to("cuda")`, không chạy CPU.
2. **Python ≥ 3.10**, torch bản CUDA, rồi `pip install -r requirements.txt`.
   Cần `tokenizers ≥ 0.21` và `transformers ≥ 5.6`; bản cũ hơn không đọc được `tokenizer.json`.
3. **Model đã merge** (Hy-MT2-1.8B + adapter, `model.safetensors` sha256 `5dc305ae…`, kèm `tokenizer.json`).
   Tạo từ adapter (sha256 `30b5fc68…`):

```bash
python tools/merge_adapter.py --adapter ADAPTER_DIR --out models/hy_mt_merged
```

## Chạy

```bash
MODEL=models/hy_mt_merged ./run.sh INPUT_DIR OUTPUT_DIR en-vi      # hoặc vi-en
```

Không có bash (Windows):

```bash
python -m docmt run-dir --input-dir IN --output-dir OUT --direction en-vi ^
    --config configs/hy_mt_vlsp26.json --run-id RUN_ID --backend hf ^
    --model C:\path\hy_mt_merged --tokenizer C:\path\hy_mt_merged\tokenizer.json
```

`INPUT_DIR` chứa các file `.md`; `OUTPUT_DIR` có cùng cây thư mục và chỉ chứa bản dịch. Log và trace nằm ở
`OUTPUT_DIR.work/` (`logs/document_status.jsonl` là trạng thái từng tài liệu). Exit code ≠ 0 khi có tài liệu
`failed` hoặc thiếu output. Biến môi trường của `run.sh`: `MODEL`, `TOKENIZER`, `CONFIG`, `RUN_ID`, `WORK_DIR`,
`PYTHON`, `BACKEND`. `--set key=value` ghi đè một trường của config.

## Chạy private test trên máy local (cần GPU)

```bash
python tools/export_sources.py --out work/sources          # dataset/private_test -> work/sources/{en,vi}/*.md
MODEL=models/hy_mt_merged ./run.sh work/sources/en work/out/vi en-vi
MODEL=models/hy_mt_merged ./run.sh work/sources/vi work/out/en vi-en
python tools/build_submission.py --en-vi work/out/vi --vi-en work/out/en --out submission.zip
```

## Docker (dịch vụ HTTP theo yêu cầu cuộc thi)

Image chạy một dịch vụ HTTP: `POST /translate` nhận một tài liệu Markdown UTF-8 và trả bản dịch, tự nhận chiều
dịch từ văn bản (nguồn tiếng Việt → `vi-en`, ngược lại `en-vi`). `GET /health` trả 200 khi model đã nạp.
Model nằm trong image, chạy offline, không gọi API ngoài.

Build từ đúng thư mục repo này (chỉ cần Docker, NVIDIA Container Toolkit và internet; không cần cài Python/torch ngoài máy):

```bash
mkdir adapter && cp /đường/dẫn/adapter_model.safetensors /đường/dẫn/adapter_config.json adapter/   # adapter không nằm trong git
tools/build_image.sh                                 # build runtime -> merge adapter + Hy-MT2-1.8B trên GPU -> build image đầy đủ
docker run --rm --gpus all -p 8000:8000 hy-mt-vlsp26
./translate.sh localhost:8000 input.md output.md     # HOST:PORT, file vào, file ra; in thống kê (giây, từ/giây, GPU)
```

`translate.sh` đợi dịch vụ sẵn sàng (tối đa `WAIT_SECONDS`, mặc định 1800 giây). `docker build --target runtime`
tạo image không có weights (mount model vào `/models/hy_mt_merged`). Chạy theo thư mục `.md` thay vì dịch vụ:
`docker run --rm --gpus all --entrypoint /app/run.sh -v IN:/input:ro -v OUT:/output hy-mt-vlsp26 /input /output en-vi`.
Cần NVIDIA Container Toolkit (`--gpus all`).

## Nộp bài (VLSP 2026 MT)

Theo trang cuộc thi, bài nộp là image Docker tự chứa chạy offline cùng script bash
`translate.sh HOST:PORT INPUT OUTPUT`; đóng gói thành một file nén, kèm MD5, đặt trên Google Drive/OneDrive
(bật quyền tải). Chỉ bản nộp cuối được chấm.

```bash
tools/package_docker.sh hy-mt-vlsp26 dist      # dist/hy-mt-vlsp26.tar.gz (image.tar + translate.sh + README.txt) và file .md5
```

Điền form nộp: Team name, Method name, Method description, Project URL (link tải archive trên), Publication URL
(link bài báo). Model 1,8B tham số (dưới giới hạn 3B). Kết quả tự động trên private test còn có thể nộp
lên AIHub dưới dạng `results.csv` (xem *Tạo submission*).

## Cấu hình

`configs/hy_mt_vlsp26.json` là config mặc định của `run.sh`:

- HF transformers, fp16, greedy, `repetition_penalty` 1.0, `batch_size` 8.
- Prompt dài tối đa 8192 token (`max_sequence`); đoạn nguồn tối đa 768 token (`source_cap`).
- Các node của một unit nối bằng xuống dòng (`extra.payload_delimiter = "newline"`), không chèn marker.
- `extra.legal_prose_blocks`: gom các đoạn văn pháp lý liền nhau thành một khối trước khi chia unit.
- Ngân sách output theo chiều: EN→VI `max(256, ⌈2.5·src⌉+128)`, VI→EN `max(256, ⌈1.25·src⌉+128)`.
- Phát hiện lặp token trên GPU (`degeneration_*`); unit bị lặp được dịch lại với `repetition_penalty` 1.15.
- Một vòng recovery cho unit lỗi; không dùng source context, glossary hay lịch sử bản dịch.

## Các bước

Luồng một tài liệu (chỉ đọc phía nguồn, không bao giờ đọc reference):

| Bước | File | Việc làm |
|---|---|---|
| s01 parse | `steps/s01_parse.py`, `s01_parse_lines.py` | raw text → `Node[]` (header, bảng, chữ ký, scope); `"".join(n.raw) == src` |
| s02 payload | `steps/s02_payload.py`, `s02_blocks.py` | chuẩn hoá payload của node; gom đoạn văn pháp lý |
| s03 split / pack | `steps/s03_split.py`, `s03_pack.py` | cắt node dài; gom node thành `Unit[]` |
| s04 prompt | `steps/s04_prompt.py` | unit → prompt |
| s05 budget | `steps/s05_budget.py` | prompt → `max_new_tokens` |
| generate | `backends/hf.py` (+ `batch_policy.py`, `degeneration.py`) | sinh theo batch trên GPU, phát hiện lặp |
| s06 parse output | `steps/s06_parse_output.py` | output → bản dịch từng node (dòng i → node i) |
| s07 validate | `steps/s07_validate_unit.py`, `checks/{numbers,headings,loops,lexical}.py` | lỗi **hard** kích hoạt recovery; warning chỉ ghi |
| s08 recovery | `steps/s08_recovery.py` | unit lỗi → dịch lại từng node |
| s09 select | `steps/s09_select.py` | chọn giữa lượt đầu và recovery |
| s10 assemble / check | `steps/s10_assemble.py`, `s10_doc_check.py` | ghép Markdown theo layout nguồn; kiểm tài liệu |

Hạ tầng: `engine/` (`doc_job.py` điều phối một tài liệu, `run.py`, `scheduler.py`, `api.py`), `core/`
(kiểu dữ liệu, config, tokenizer, normalize, hashing), `inputs/sources.py`, `tracing/` (trace, status, cache),
`cli.py` (lệnh `run-dir`), `server.py` (dịch vụ HTTP của image Docker).

## Chạy trên Kaggle (private test, T4 x2)

Thư mục `kaggle/`:

| File | Tác dụng |
|---|---|
| `kernel_private.py` | Kernel Kaggle: merge adapter vào Hy-MT2-1.8B, chạy `run-dir` song song hai chiều (mỗi chiều một GPU), ghi `predictions_private/{en,vi}/`, `submission.zip` và `results.zip` |
| `build_bundle.py` | Đóng gói code, `calibration/`, CSV private và `run_spec.json` thành dataset runtime kèm manifest sha256 mà kernel kiểm tra |
| `push.sh` | Dựng bundle, upload dataset runtime + adapter, push kernel |
| `calibration/{en-vi,vi-en}/config.json` | Config từng chiều (batch 8) mà kernel truyền cho `run-dir` |
| `run_spec.json` | Thông số run; `git_head` và `csv_md5` được điền khi build |

```bash
KAGGLE_API_TOKEN=... OWNER=<user kaggle> ADAPTER_DIR=<thư mục adapter> kaggle/push.sh
```

Adapter cần có sha256 `30b5fc68…`; kernel dừng nếu khác. CSV private mặc định là
`dataset/private_test/private_test_input.csv` (đổi bằng `PRIVATE_CSV`). Kernel cần 2 GPU T4 và internet
(tải `tencent/Hy-MT2-1.8B` đúng revision đã ghim). Chạy xong lấy kết quả bằng
`kaggle kernels output OWNER/KERNEL -p OUT`; `OUT/submission.zip` là file nộp. `kaggle/build/` là thư mục tạm.

## Công cụ (`tools/`)

| File | Tác dụng |
|---|---|
| `build_submission.py` | Đóng gói bản dịch thành `results.csv` trong ZIP để nộp |
| `score_public.py` | Chấm BLEU/chrF trên public test (`dataset/public_test/en-vi-public_test.csv`) |
| `calibrate_batch.py` | Đo throughput GPU để chọn batch 8 hoặc 16 |
| `merge_adapter.py` | Merge LoRA adapter vào Hy-MT2-1.8B |
| `package_docker.sh` | Đóng gói image + `translate.sh` thành archive kèm MD5 để nộp |
| `export_sources.py` | Tách CSV private thành thư mục `.md` làm input cho `run.sh` |

## Tạo submission

```bash
python tools/build_submission.py --en-vi OUTPUT_EN_VI --vi-en OUTPUT_VI_EN --out submission.zip
```

Template mặc định là `dataset/private_test/private_test_input.csv`: mỗi ID một chiều (nguồn `en` → `en-vi`,
nguồn `vi` → `vi-en`). Cũng nhận template có cột `sample_id,direction,translation`. Thư mục `--en-vi`
chứa bản dịch sang tiếng Việt, `--vi-en` chứa bản dịch sang tiếng Anh, mỗi thư mục giữ cây `category/id.md`.
Script báo lỗi và không tạo ZIP nếu thiếu file, bản dịch rỗng, key trùng hoặc direction sai.

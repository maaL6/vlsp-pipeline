# VLSP 26 - Document-Level Legal Machine Translation with Limited-Parameter Models

Our submission to the VLSP 2026 machine translation task translates legal Markdown documents in both directions, EN→VI and VI→EN. It uses Hy-MT2-1.8B with a merged LoRA adapter. Each processing step lives in its own file (`docmt/steps/`). For building, running and submitting the Docker image, see [GUIDE.md](GUIDE.md).

Public test results (100 documents, raw scoring with `tools/score_public.py`):

| Direction | BLEU | chrF |
|---|---|---|
| EN→VI | 70.87 | 84.48 |
| VI→EN | 47.44 | 79.50 |

Private test leaderboard (our team, wann, ranks 3rd on both metrics):

| # | User | Entries | Date of last entry | SacreBLEU | chrF |
|---|---|---|---|---|---|
| 1 | NgJaBach | 5 | 10/11/26 | 63.5312 | 84.1597 |
| 2 | rudeus | 5 | 10/14/26 | 62.1307 | 83.2732 |
| 3 | **wann** | 2 | 10/14/26 | 60.4346 | 82.1852 |
| 4 | An0949033378 | 6 | 10/14/26 | 59.7272 | 81.9513 |
| 5 | hiimhoanglam | 2 | 10/11/26 | 57.6959 | 80.5156 |
| 6 | phthuan05 | 1 | 10/11/26 | 55.7442 | 80.4560 |
| 7 | bao135 | 6 | 10/13/26 | 52.9761 | 77.5254 |

## Setup

The repository does not contain the model. The `.safetensors` files are available in the Releases section of this repository. You need:

1. An NVIDIA GPU (~5 GB VRAM, fp16; tested on a T4). The `hf` backend calls `.to("cuda")` and does not run on CPU.
2. Python ≥ 3.10 and a CUDA build of torch, then `pip install -r requirements.txt`. `tokenizers ≥ 0.21` and `transformers ≥ 5.6` are required; older versions cannot read `tokenizer.json`.
3. The merged model (Hy-MT2-1.8B + adapter, `model.safetensors` sha256 `5dc305ae…`, with `tokenizer.json`). Build it from the adapter (sha256 `30b5fc68…`):

```bash
python tools/merge_adapter.py --adapter ADAPTER_DIR --out models/hy_mt_merged
```

## Running

```bash
MODEL=models/hy_mt_merged ./run.sh INPUT_DIR OUTPUT_DIR en-vi      # or vi-en
```

Without bash (Windows):

```bash
python -m docmt run-dir --input-dir IN --output-dir OUT --direction en-vi ^
    --config configs/hy_mt_vlsp26.json --run-id RUN_ID --backend hf ^
    --model C:\path\hy_mt_merged --tokenizer C:\path\hy_mt_merged\tokenizer.json
```

`INPUT_DIR` holds the `.md` files. `OUTPUT_DIR` mirrors its directory tree and contains only translations. Logs and traces go to `OUTPUT_DIR.work/`, where `logs/document_status.jsonl` records the status of each document. The exit code is non-zero if any document is `failed` or an output is missing. `run.sh` reads these environment variables: `MODEL`, `TOKENIZER`, `CONFIG`, `RUN_ID`, `WORK_DIR`, `PYTHON`, `BACKEND`. `--set key=value` overrides a single config field.

## Private test and submission

```bash
python tools/export_sources.py --out work/sources          # dataset/private_test -> work/sources/{en,vi}/*.md
MODEL=models/hy_mt_merged ./run.sh work/sources/en work/out/vi en-vi
MODEL=models/hy_mt_merged ./run.sh work/sources/vi work/out/en vi-en
python tools/build_submission.py --en-vi work/out/vi --vi-en work/out/en --out submission.zip
```

The default template is `dataset/private_test/private_test_input.csv`, where each ID has one direction (`en` source → `en-vi`, `vi` source → `vi-en`). A template with the columns `sample_id,direction,translation` also works. The `--en-vi` and `--vi-en` directories each keep the `category/id.md` tree. The script reports an error and writes no ZIP if a file is missing, a translation is empty, a key is duplicated or a direction is wrong.

## Docker

The image runs an HTTP service as the task requires. `POST /translate` takes one UTF-8 Markdown document and returns its translation, detecting the direction from the text (Vietnamese source → `vi-en`, otherwise `en-vi`). `GET /health` returns 200 once the model is loaded. The model is inside the image and runs offline.

```bash
mkdir adapter && cp /path/to/adapter_model.safetensors /path/to/adapter_config.json adapter/   # the adapter is not in git
tools/build_image.sh
docker run --rm --gpus all -p 8000:8000 hy-mt-vlsp26
./translate.sh localhost:8000 input.md output.md
./translate.sh localhost:8000 dataset/private_test/private_test_input.csv results.csv
```

Docker and the NVIDIA Container Toolkit are required. `translate.sh` waits for the service for up to `WAIT_SECONDS` (default 1800). To package the image for submission (archive + MD5):

```bash
tools/package_docker.sh hy-mt-vlsp26 dist
```

Other ways to run the image (without HTTP, over a directory of `.md` files, or without weights) and the submission form fields are in [GUIDE.md](GUIDE.md).

## Configuration

`configs/hy_mt_vlsp26.json` is the default config for `run.sh`:

- HF transformers, fp16, greedy decoding, `repetition_penalty` 1.0, `batch_size` 8.
- Prompts are capped at 8192 tokens (`max_sequence`) and source segments at 768 tokens (`source_cap`).
- The nodes of a unit are joined by newlines (`extra.payload_delimiter = "newline"`), with no markers inserted.
- `extra.legal_prose_blocks` merges adjacent legal prose paragraphs into one block before units are split.
- Output budget: EN→VI `max(256, ⌈2.5·src⌉+128)`, VI→EN `max(256, ⌈1.25·src⌉+128)`.
- Token repetition is detected on the GPU (`degeneration_*`), and a degenerate unit is retranslated with `repetition_penalty` 1.15.
- One recovery pass for failed units. No source context, glossary or translation history is used.

## Pipeline steps

The pipeline reads only the source side and never the reference:

| Step | File | Role |
|---|---|---|
| s01 parse | `steps/s01_parse.py`, `s01_parse_lines.py` | raw text → `Node[]` (header, table, signature, scope); `"".join(n.raw) == src` |
| s02 payload | `steps/s02_payload.py`, `s02_blocks.py` | normalize node payloads; merge legal prose paragraphs |
| s03 split / pack | `steps/s03_split.py`, `s03_pack.py` | split long nodes; pack nodes into `Unit[]` |
| s04 prompt | `steps/s04_prompt.py` | unit → prompt |
| s05 budget | `steps/s05_budget.py` | prompt → `max_new_tokens` |
| generate | `backends/hf.py` (+ `batch_policy.py`, `degeneration.py`) | batched GPU generation, repetition detection |
| s06 parse output | `steps/s06_parse_output.py` | output → per-node translations (line i → node i) |
| s07 validate | `steps/s07_validate_unit.py`, `checks/{numbers,headings,loops,lexical}.py` | hard errors trigger recovery; warnings are only logged |
| s08 recovery | `steps/s08_recovery.py` | failed unit → retranslate node by node |
| s09 select | `steps/s09_select.py` | choose between the first pass and recovery |
| s10 assemble / check | `steps/s10_assemble.py`, `s10_doc_check.py` | rebuild Markdown in the source layout; check the document |

Infrastructure: `engine/` (`doc_job.py`, `run.py`, `scheduler.py`, `api.py`), `core/` (data types, config, tokenizer, normalize, hashing), `inputs/sources.py`, `tracing/` (trace, status, cache), `cli.py` (the `run-dir` command), `server.py` (the Docker image's HTTP service).

## Kaggle (private test, 2x T4)

The `kaggle/` directory contains:

| File | Role |
|---|---|
| `kernel_private.py` | Merges the adapter into Hy-MT2-1.8B, runs `run-dir` on both directions in parallel (one GPU each), and writes `predictions_private/{en,vi}/`, `submission.zip` and `results.zip` |
| `build_bundle.py` | Packs the code, `calibration/`, the private CSV and `run_spec.json` into a runtime dataset with a sha256 manifest |
| `push.sh` | Builds the bundle, uploads the runtime dataset and the adapter, and pushes the kernel |
| `calibration/{en-vi,vi-en}/config.json` | Per-direction config (batch 8) |
| `run_spec.json` | Run parameters; `git_head` and `csv_md5` are filled in at build time |

```bash
KAGGLE_API_TOKEN=... OWNER=<kaggle user> ADAPTER_DIR=<adapter directory> kaggle/push.sh
```

The adapter must have sha256 `30b5fc68…`, otherwise the kernel stops. The private CSV defaults to `dataset/private_test/private_test_input.csv` (override with `PRIVATE_CSV`). The kernel needs 2 T4 GPUs and internet access to download `tencent/Hy-MT2-1.8B` at the pinned revision. Fetch the results with `kaggle kernels output OWNER/KERNEL -p OUT`; `OUT/submission.zip` is the file to submit. `kaggle/build/` is a temporary directory.

## Tools (`tools/`)

| File | Role |
|---|---|
| `build_submission.py` | Packs translations into `results.csv` inside a ZIP for submission |
| `score_public.py` | Scores BLEU/chrF on the public test (`dataset/public_test/en-vi-public_test.csv`) |
| `calibrate_batch.py` | Measures GPU throughput to choose batch size 8 or 16 |
| `merge_adapter.py` | Merges the LoRA adapter into Hy-MT2-1.8B |
| `compare_csv.py` | Compares a result CSV with an expected CSV (matching rows, chrF); used by the demo in `examples/` |
| `package_docker.sh` | Packs the image and `translate.sh` into an archive with an MD5 |
| `export_sources.py` | Splits the private CSV into `.md` directories as input for `run.sh` |

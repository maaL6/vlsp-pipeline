# Guide: build, run and submit the Docker image (VLSP 2026 MT)

## Quick start

Needs a Linux machine with an NVIDIA GPU, Docker, NVIDIA Container Toolkit and internet.

```bash
git clone https://github.com/maaL6/vlsp-pipeline && cd vlsp-pipeline
gh auth login                                  # to download the adapter from release adapter-v1 (or put the files in adapter/ yourself)
tools/build_image.sh                           # merge the adapter + build image hy-mt-vlsp26 (model included, runs offline)

# Demo: 4 short private documents, takes a few minutes
mkdir -p out
docker run --rm --gpus all -v $PWD/examples:/in:ro -v $PWD/out:/out hy-mt-vlsp26 --csv /in/demo_input.csv --out /out/demo_results.csv
pip install sacrebleu
python3 tools/compare_csv.py out/demo_results.csv examples/demo_expected_kaggle.csv --min-chrf 90    # PASS if it matches the Kaggle output

# Full private test (several hours)
docker run --rm --gpus all -v $PWD/dataset/private_test:/in:ro -v $PWD/out:/out hy-mt-vlsp26 --csv /in/private_test_input.csv --out /out/results.csv

# Package for submission (archive + MD5)
tools/package_docker.sh hy-mt-vlsp26 dist
```

The demo output should have no empty rows and a high chrF against `examples/demo_expected_kaggle.csv` (the Kaggle penalty100 run). Many rows match byte for byte; small differences come from the torch version or GPU.

## 1. Task requirements

Per the [competition page](https://vlsp.org.vn/vlsp2026/eval/mt_challenge), a submission is:

- a self-contained Docker image that runs offline with no external API calls;
- a bash script taking `host:port`, an input file and an output file, which sends the document to the image and writes the UTF-8 Markdown translation (printing time and words per second is encouraged);
- both in an archive with an MD5, hosted on Google Drive or OneDrive with download permission;
- a form with Team name, Method name, Method description, Project URL (download link) and Publication URL (paper link).

The organizers run the image as a web service and call the script on each private document (Vietnamese or English only). Scoring covers SacreBLEU, terminology, legal accuracy, inference time and peak memory. The model must have fewer than 3B parameters (ours has 1.8B). The page does not say how the container is started, so the image listens on port `8000` (set `PORT` to change it) and `translate.sh` accepts any `host:port`.

## 2. Requirements

1. NVIDIA driver 570 or later (CUDA 12.8) with a working `nvidia-smi`. Older drivers may work but are untested.
2. Docker Engine and the NVIDIA Container Toolkit (Ubuntu/Debian):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi    # check that Docker sees the GPU
```

3. About 20 GB of free disk and access to `registry-1.docker.io`, `download.pytorch.org`, `pypi.nvidia.com`, `pypi.org` and `huggingface.co` (the merge downloads `tencent/Hy-MT2-1.8B`). After the build, the image runs offline.
4. A logged-in `gh` to download the adapter from the GitHub Release, or the adapter files copied into `adapter/`.

## 3. Build

```bash
tools/build_image.sh            # the image is named hy-mt-vlsp26; the first argument changes the name
```

Without `gh`, put `adapter_model.safetensors` (sha256 `30b5fc68…`) and `adapter_config.json` in `adapter/` first. The script builds the `runtime` image, merges the adapter into Hy-MT2-1.8B inside a container (output in `models/hy_mt_merged/`, skipped if present), then builds the `full` image with the model.

The merge files are created by root. On Kaggle the merged `model.safetensors` sha256 starts with `5dc305ae`; another GPU or torch version can differ by a few bits, so a different hash is not an error. The tests in section 5 decide.

## 4. Run

**Private test (CSV in, CSV out).** `private_test_input.csv` has the columns `sample_id, category, source_language, source_markdown`. `results.csv` has `sample_id, direction, translation`, in the same order with every row (`en` → `en-vi`, `vi` → `vi-en`). A document that fails twice is copied from the source and logged.

```bash
# Option 1: directly, no HTTP (recommended for the full set)
docker run --rm --gpus all -v $PWD/dataset/private_test:/in:ro -v $PWD/out:/out hy-mt-vlsp26 \
    --csv /in/private_test_input.csv --out /out/results.csv
# Option 2: through the HTTP service
docker run -d --name hy --gpus all -p 8000:8000 hy-mt-vlsp26
./translate.sh localhost:8000 dataset/private_test/private_test_input.csv results.csv
```

**One document:**

```bash
docker run --rm --gpus all -p 8000:8000 hy-mt-vlsp26       # logs "listening on 0.0.0.0:8000" when ready
./translate.sh localhost:8000 input.md output.md
```

`translate.sh` waits up to `WAIT_SECONDS` (default 1800) for the service, then prints `direction`, `input_words`, `seconds`, `words_per_second` and `peak_gpu_memory_mb`. The direction is detected from the text (Vietnamese source → `vi-en`, otherwise `en-vi`).

**HTTP API:**

```bash
curl -s localhost:8000/health                                              # {"status": "ok"} once the model is loaded
curl -s -X POST --data-binary @input.md localhost:8000/translate -o output.md
curl -s -X POST --data-binary @input.md "localhost:8000/translate?direction=en-vi" -o output.md   # force the direction
```

The service handles one request at a time. Errors return JSON `{"error": ...}` with status 400 (empty document, not UTF-8, invalid direction) or 500 (translation failed). To change the port, use `-e PORT=9000 -p 9000:9000`.

**Directory of `.md` files:**

```bash
docker run --rm --gpus all --entrypoint /app/run.sh -v $PWD/in:/input:ro -v $PWD/out:/output hy-mt-vlsp26 /input /output en-vi
```

`python3 tools/export_sources.py --out work/sources` splits the private CSV into `work/sources/{en,vi}/*.md`. To zip the result, use `zip submission.zip results.csv` or `tools/build_submission.py`.

## 5. Tests

With the image built and the service running (`docker run -d --name hy --gpus all -p 8000:8000 hy-mt-vlsp26`):

1. GPU and torch: `docker run --rm --gpus all --entrypoint python hy-mt-vlsp26 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` should print `2.11.0+cu128 True <GPU name>`.
2. Model in the image: `docker run --rm --entrypoint ls hy-mt-vlsp26 -la /models/hy_mt_merged` lists `model.safetensors`, `tokenizer.json` and `config.json`.
3. Health: wait for `[server] listening on 0.0.0.0:8000` in `docker logs -f hy`, then `curl -s localhost:8000/health`. `docker ps` shows `(healthy)` once the model is loaded.
4. Submission script: `./translate.sh localhost:8000 input.md output.md`.
5. Public data with references:
   ```bash
   pip install sacrebleu
   python3 tools/test_service.py --addr localhost:8000 --n 3        # the 3 shortest documents, both directions
   ```
   Each request must return HTTP 200 with the detected direction and a non-empty body. The script prints BLEU/chrF against the reference. `--n 0` runs all 50 documents (several hours). Kaggle run scores: EN→VI BLEU 70.87 / chrF 84.48, VI→EN BLEU 47.44 / chrF 79.50.
6. Against the Kaggle penalty100 submission: `python3 tools/test_service.py --addr localhost:8000 --mode private --n 3 --min-chrf 90` prints the number of byte-identical outputs and the chrF. Most should match or be very close; batching and torch/GPU differences can change a few sentences.
7. Offline: start the container with `--network none`, then `docker cp input.md hy:/tmp/in.md`, `docker exec hy /app/translate.sh localhost:8000 /tmp/in.md /tmp/out.md` and `docker cp hy:/tmp/out.md output.md`.
8. Shutdown: `time docker stop hy` should take a few seconds. Measure resources with `nvidia-smi` or `docker stats`.
9. Package on a clean machine: `tar xzf hy-mt-vlsp26.tar.gz && docker load -i image.tar`, then repeat steps 3 and 4.

Without a GPU (very slow, about 8 GB RAM, only checks the data path): `docker run --rm -p 8000:8000 -e HF_DEVICE=cpu -e HF_DTYPE=float32 hy-mt-vlsp26`.

## 6. Package and submit

```bash
tools/package_docker.sh hy-mt-vlsp26 dist
```

This creates `dist/hy-mt-vlsp26.tar.gz` (with `image.tar`, `translate.sh` and `README.md`) and `dist/hy-mt-vlsp26.tar.gz.md5`. Upload the archive to Google Drive or OneDrive with download permission, then paste the link and the MD5 into the form. The recipient runs `docker load -i image.tar` and follows section 4.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | Install and configure the NVIDIA Container Toolkit (section 2) |
| `put adapter_model.safetensors ... in adapter/` | Run `gh auth login` and retry, or copy the files into `adapter/` |
| `adapter sha256 differs` | Wrong adapter; use the `adapter-v1` release (sha `30b5fc68…`) |
| Download of `tencent/Hy-MT2-1.8B` fails | Merge on a machine that can reach HuggingFace and place the result in `models/hy_mt_merged/` |
| `CUDA out of memory` | Free at least ~5 GB of GPU memory |
| `permission denied` deleting `models/hy_mt_merged` | Created by root: `sudo rm -rf models/hy_mt_merged` |
| `.sh` reports `bad interpreter` / `\r` | CRLF from a Windows checkout; clone again or run `git add --renormalize .` |
| `ReadTimeoutError ... pypi.nvidia.com` or `context deadline exceeded` | Slow or blocked network; rerun `tools/build_image.sh` (finished layers are reused, pip retries 10 times with a 300 s timeout) |
| Build runs out of disk | Free about 20 GB; `docker system prune` clears the cache |
| `service ... is not ready` | The model is still loading or `host:port` is wrong; check the container logs |

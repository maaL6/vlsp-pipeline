# Hướng dẫn: build, chạy, nộp Docker image (VLSP 2026 MT)

## Tóm tắt: build, chạy, demo (máy Linux có GPU, Docker, NVIDIA Container Toolkit, internet)

```bash
git clone https://github.com/maaL6/vlsp-pipeline && cd vlsp-pipeline
gh auth login                                  # để tải adapter từ release adapter-v1 (hoặc tự đặt file vào adapter/)
tools/build_image.sh                           # merge adapter + build image hy-mt-vlsp26 (có sẵn model, chạy offline)

# Demo nhanh: 4 tài liệu private ngắn (2 en, 2 vi), mất vài phút
mkdir -p out
docker run --rm --gpus all -v $PWD/examples:/in:ro -v $PWD/out:/out hy-mt-vlsp26 --csv /in/demo_input.csv --out /out/demo_results.csv
pip install sacrebleu
python3 tools/compare_csv.py out/demo_results.csv examples/demo_expected_kaggle.csv --min-chrf 90    # PASS nếu khớp bản Kaggle

# Chạy thật toàn bộ private test (nhiều giờ)
docker run --rm --gpus all -v $PWD/dataset/private_test:/in:ro -v $PWD/out:/out hy-mt-vlsp26 --csv /in/private_test_input.csv --out /out/results.csv

# Đóng gói nộp (archive + MD5), rồi upload lên Drive/OneDrive và điền form
tools/package_docker.sh hy-mt-vlsp26 dist
```

`examples/demo_input.csv` là 4 tài liệu private ngắn nhất; `examples/demo_expected_kaggle.csv` là bản dịch tương ứng của lượt Kaggle penalty100.
Kỳ vọng: không dòng nào rỗng và chrF so với bản Kaggle cao (nhiều dòng trùng từng byte; có thể lệch nhẹ do khác phiên bản torch/GPU).
Cũng chạy được qua HTTP: `docker run -d --gpus all -p 8000:8000 hy-mt-vlsp26` rồi `./translate.sh localhost:8000 examples/demo_input.csv out/demo_results.csv`.
Chi tiết từng bước bên dưới.

## 1. Đề bài yêu cầu gì, image được dùng ra sao

Theo trang cuộc thi (<https://vlsp.org.vn/vlsp2026/eval/mt_challenge>), bài nộp gồm:

- một **Docker image tự chứa** (có đủ model và thư viện, chạy **offline**, không gọi API ngoài);
- một **script bash** nhận 3 tham số: `host:port` của image, file đầu vào, file đầu ra. Script gửi file tới image
  và ghi bản dịch (Markdown UTF-8, đủ cả tài liệu) ra file đầu ra. In thống kê (thời gian, từ/giây) là khuyến khích;
- các file trên nén (`.tar.gz`, `.zip`, ...) kèm **MD5**, đặt trên Google Drive/OneDrive có quyền tải;
- form nộp có: Team name, Method name, Method description, Project URL (link tải image), Publication URL (link bài báo).

Ban tổ chức sẽ nạp image, chạy nó thành một dịch vụ web rồi gọi script với từng tài liệu private test (mỗi
tài liệu chỉ có tiếng Việt hoặc chỉ có tiếng Anh). Điểm gồm SacreBLEU, thuật ngữ, độ chính xác pháp lý, **thời gian
suy luận** và **bộ nhớ đỉnh**. Model phải dưới 3B tham số (đây là 1,8B).

Trang không nêu cụ thể cách ban tổ chức khởi động container (cổng, cờ GPU, timeout). Image này lắng nghe cổng `8000`
(đổi bằng biến `PORT`) và `translate.sh` nhận bất kỳ `host:port` nào.

## 2. Chuẩn bị máy GPU (Linux)

1. Driver NVIDIA: khuyến nghị bản 570 trở lên (CUDA 12.8), `nvidia-smi` chạy được. Driver cũ hơn có thể vẫn chạy nhờ tương thích minor-version nhưng chưa được kiểm.
2. Docker Engine.
3. NVIDIA Container Toolkit (Ubuntu/Debian):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

4. Kiểm tra Docker thấy GPU: `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`.
5. Internet và khoảng 20 GB ổ trống. Build cần truy cập `registry-1.docker.io`, `download.pytorch.org`, `pypi.nvidia.com` (torch lấy thư viện CUDA từ đây) và `pypi.org`; bước merge cần thêm `huggingface.co` (tải `tencent/Hy-MT2-1.8B`). Sau khi build xong, image chạy hoàn toàn offline.
6. `gh` đã đăng nhập (`gh auth login`) nếu muốn tải adapter từ GitHub Release; hoặc tự copy adapter vào `adapter/`.

## 3. Build image (chỉ cần folder repo)

```bash
git clone https://github.com/maaL6/vlsp-pipeline && cd vlsp-pipeline
tools/build_image.sh            # image tên hy-mt-vlsp26; tham số đầu tiên đổi tên image
```

Nếu không dùng `gh`, tự đặt adapter (`adapter_model.safetensors` sha256 `30b5fc68…` và `adapter_config.json`) vào `adapter/`
trước khi chạy. Script làm 3 việc: build image `runtime` (chưa có model) → merge adapter vào Hy-MT2-1.8B trong
container, ghi ra `models/hy_mt_merged/` (bỏ qua nếu đã có) → build image `full` có sẵn model.

Model merge ra thư mục `models/hy_mt_merged/` (file do root trong container tạo; xoá bằng `sudo rm -rf` nếu cần). Bản merge trên
Kaggle có sha256 `model.safetensors` bắt đầu bằng `5dc305ae`; trên GPU hoặc phiên bản torch khác, kết quả phép cộng LoRA có thể lệch vài
bit nên **hash khác không có nghĩa là lỗi**. Tiêu chí đúng/sai là các bài kiểm tra ở mục 5.

## 4. Chạy và sử dụng

**Private test (1 file CSV vào, 1 file CSV ra):** `dataset/private_test/private_test_input.csv` (cột `sample_id, category, source_language, source_markdown`)
thành file nộp `results.csv` (cột `sample_id, direction, translation`, đúng thứ tự, đủ mọi dòng; `en` → `en-vi`, `vi` → `vi-en`).
Tài liệu nào lỗi hai lần thì chép nguyên văn nguồn (có ghi log) để file nộp luôn đủ dòng.

```bash
# Cách 1: chạy thẳng, không cần HTTP (khuyến nghị cho cả bộ private, mất nhiều giờ)
docker run --rm --gpus all -v $PWD/dataset/private_test:/in:ro -v $PWD/out:/out hy-mt-vlsp26 \
    --csv /in/private_test_input.csv --out /out/results.csv
# Cách 2: qua dịch vụ HTTP
docker run -d --name hy --gpus all -p 8000:8000 hy-mt-vlsp26
./translate.sh localhost:8000 dataset/private_test/private_test_input.csv results.csv
```

Nén file nộp nếu cần: `zip submission.zip results.csv` (hoặc dùng `tools/build_submission.py` với các thư mục `.md`).

**Một tài liệu Markdown:**

```bash
docker run --rm --gpus all -p 8000:8000 hy-mt-vlsp26       # nạp model, log "listening on 0.0.0.0:8000" khi sẵn sàng
./translate.sh localhost:8000 input.md output.md           # dịch một tài liệu, in thống kê
```

`translate.sh` đợi dịch vụ sẵn sàng (tối đa `WAIT_SECONDS`, mặc định 1800 giây), gửi file, ghi bản dịch và in:
`direction`, `input_words`, `seconds`, `words_per_second`, `peak_gpu_memory_mb`. Chiều dịch tự nhận từ văn bản
(nguồn tiếng Việt → `vi-en`, ngược lại `en-vi`).

Gọi trực tiếp bằng HTTP:

```bash
curl -s localhost:8000/health                                              # {"status": "ok"} khi model đã nạp
curl -s -X POST --data-binary @input.md localhost:8000/translate -o output.md
curl -s -X POST --data-binary @input.md "localhost:8000/translate?direction=en-vi" -o output.md   # ép chiều dịch
```

Dịch vụ xử lý từng yêu cầu một (một GPU). Lỗi trả JSON `{"error": ...}` với mã 400 (tài liệu rỗng, không phải UTF-8,
chiều sai) hoặc 500 (dịch thất bại). Đổi cổng: `-e PORT=9000 -p 9000:9000`.

Chạy theo thư mục `.md` thay vì dịch vụ:

```bash
docker run --rm --gpus all --entrypoint /app/run.sh -v $PWD/in:/input:ro -v $PWD/out:/output hy-mt-vlsp26 /input /output en-vi
```

Thử với dữ liệu có sẵn: `python3 tools/export_sources.py --out work/sources` tách `dataset/private_test/private_test_input.csv`
thành `work/sources/{en,vi}/*.md`.

## 5. Kiểm thử image

Chạy lần lượt (mỗi bước cần image `hy-mt-vlsp26` đã build; bước 4 trở đi cần container đang chạy).

1. **Image thấy GPU và đúng phiên bản torch**
   ```bash
   docker run --rm --gpus all --entrypoint python hy-mt-vlsp26 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
   Kỳ vọng: `2.11.0+cu128 True <tên GPU>`.
2. **Model nằm trong image:** `docker run --rm --entrypoint ls hy-mt-vlsp26 -la /models/hy_mt_merged` (có `model.safetensors`, `tokenizer.json`, `config.json`).
3. **Khởi động và tự kiểm sức khoẻ**
   ```bash
   docker run -d --name hy --gpus all -p 8000:8000 hy-mt-vlsp26
   docker logs -f hy                      # đợi dòng "[server] listening on 0.0.0.0:8000"
   curl -s localhost:8000/health          # {"status": "ok"}
   docker ps                              # cột STATUS chuyển sang (healthy) sau khi model nạp xong
   ```
4. **Dịch thử một tài liệu bằng đúng script nộp bài:** `./translate.sh localhost:8000 input.md output.md`
   (đọc output, kiểm tra không còn lỗi, có số liệu thống kê).
5. **Kiểm thử tự động với dữ liệu public (có reference)**
   ```bash
   pip install sacrebleu
   python3 tools/test_service.py --addr localhost:8000 --n 3        # 3 tài liệu ngắn nhất, cả hai chiều
   ```
   Mỗi yêu cầu phải trả HTTP 200, đúng chiều dịch tự nhận, không rỗng; in BLEU/chrF so với reference. `--n 0` chạy cả 50 tài liệu
   (nhiều giờ); điểm toàn bộ tham chiếu của bản Kaggle: EN→VI BLEU 70.87 / chrF 84.48, VI→EN BLEU 47.44 / chrF 79.50.
6. **Đối chiếu với bản nộp penalty100 trên Kaggle**
   ```bash
   python3 tools/test_service.py --addr localhost:8000 --mode private --n 3 --min-chrf 90
   ```
   So kết quả của dịch vụ với `submission.zip` cho các tài liệu private ngắn nhất: in số bản trùng từng byte và chrF giữa hai bên.
   Phần lớn nên trùng hoặc rất gần; vài câu có thể khác vì dịch từng tài liệu một (khác cách ghép batch) và khác phiên bản torch/GPU.
7. **Chạy offline thật sự** (không mạng): dùng `curl` có sẵn trong image
   ```bash
   docker rm -f hy; docker run -d --name hy --gpus all --network none hy-mt-vlsp26
   docker cp input.md hy:/tmp/in.md
   docker exec hy /app/translate.sh localhost:8000 /tmp/in.md /tmp/out.md
   docker cp hy:/tmp/out.md output.md
   ```
8. **Tắt gọn:** `time docker stop hy` phải xong trong vài giây (dịch vụ bắt SIGTERM). Đo tài nguyên bằng `nvidia-smi`/`docker stats`.
9. **Kiểm tra gói nộp trên máy sạch:** `tar xzf hy-mt-vlsp26.tar.gz && docker load -i image.tar`, rồi lặp lại các bước 3–4 (hoặc `docker rmi hy-mt-vlsp26` rồi `docker load` ngay trên máy này).

Thử khung dịch vụ khi không có GPU (rất chậm, cần khoảng 8 GB RAM, chỉ để kiểm tra đường đi của dữ liệu):
`docker run --rm -p 8000:8000 -e HF_DEVICE=cpu -e HF_DTYPE=float32 hy-mt-vlsp26`.

## 6. Đóng gói và nộp

```bash
tools/package_docker.sh hy-mt-vlsp26 dist
```

Tạo `dist/hy-mt-vlsp26.tar.gz` (gồm `image.tar`, `translate.sh`, `README.md`) và `dist/hy-mt-vlsp26.tar.gz.md5`.
Upload file `.tar.gz` lên Google Drive/OneDrive, bật quyền tải, dán link vào ô Project URL cùng MD5 theo yêu cầu form.
Người nhận dùng: `docker load -i image.tar`, rồi chạy như mục 4.

## 7. Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `could not select device driver "" with capabilities: [[gpu]]` | Chưa cài/cấu hình NVIDIA Container Toolkit (mục 2) |
| `put adapter_model.safetensors ... in adapter/` | Thiếu adapter: `gh auth login` rồi chạy lại, hoặc copy file vào `adapter/` |
| Merge báo `adapter sha256 differs` | Sai file adapter; dùng đúng bản `adapter-v1` (sha `30b5fc68…`) |
| Lỗi tải `tencent/Hy-MT2-1.8B` khi merge | Máy không ra được HuggingFace; merge ở máy khác rồi đặt `models/hy_mt_merged/` |
| `CUDA out of memory` | Cần GPU ≥ ~5 GB trống; đóng tiến trình khác dùng GPU |
| `permission denied` khi xoá `models/hy_mt_merged` | File do root trong container tạo: `sudo rm -rf models/hy_mt_merged` |
| Script `.sh` báo `bad interpreter` / `\r` | Checkout trên Windows đổi sang CRLF; `.gitattributes` đã ép LF, clone lại hoặc `git add --renormalize .` |
| Build lỗi `ReadTimeoutError ... pypi.nvidia.com` (hoặc `context deadline exceeded` khi kéo image) | Mạng chậm hoặc bị chặn; chạy lại `tools/build_image.sh` (các layer đã xong được dùng lại, pip đã đặt timeout 300 giây và thử lại 10 lần) |
| Build hết dung lượng | Cần khoảng 20 GB trống; `docker system prune` để dọn cache |
| `translate.sh` báo `service ... is not ready` | Model đang nạp hoặc sai `host:port`; xem log container |

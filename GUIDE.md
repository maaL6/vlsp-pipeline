# Hướng dẫn: build, chạy, nộp Docker image (VLSP 2026 MT)

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

1. Driver NVIDIA hỗ trợ CUDA 12.8 (bản 570 trở lên): `nvidia-smi` chạy được.
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

4. Kiểm tra Docker thấy GPU: `docker run --rm --gpus all nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 nvidia-smi`.
5. Internet (bước merge tải `tencent/Hy-MT2-1.8B` từ HuggingFace; sau đó image chạy offline) và khoảng 20 GB ổ trống.
6. `gh` đã đăng nhập (`gh auth login`) nếu muốn tải adapter từ GitHub Release; hoặc tự copy adapter vào `adapter/`.

## 3. Build image (chỉ cần folder repo)

```bash
git clone https://github.com/maaL6/vlsp-pipeline && cd vlsp-pipeline
tools/build_image.sh            # image tên hy-mt-vlsp26; tham số đầu tiên đổi tên image
```

Nếu không dùng `gh`, tự đặt adapter (`adapter_model.safetensors` sha256 `30b5fc68…` và `adapter_config.json`) vào `adapter/`
trước khi chạy. Script làm 3 việc: build image `runtime` (chưa có model) → merge adapter vào Hy-MT2-1.8B trong
container, ghi ra `models/hy_mt_merged/` (bỏ qua nếu đã có) → build image `full` có sẵn model.

Kiểm tra model merge đúng: `sha256sum models/hy_mt_merged/model.safetensors` phải bắt đầu bằng `5dc305ae`.

## 4. Chạy và sử dụng

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

## 5. Kiểm tra chất lượng (khuyến nghị trước khi nộp)

Dịch các tài liệu public qua dịch vụ rồi chấm bằng `tools/score_public.py` (cần `pip install sacrebleu`):
kết quả tham chiếu của bản chạy Kaggle là EN→VI BLEU 70.87 / chrF 84.48, VI→EN BLEU 47.44 / chrF 79.50.
Chạy từng tài liệu có thể lệch nhẹ so với chạy hàng loạt (khác cách ghép batch, phiên bản torch, GPU).

## 6. Đóng gói và nộp

```bash
tools/package_docker.sh hy-mt-vlsp26 dist
```

Tạo `dist/hy-mt-vlsp26.tar.gz` (gồm `image.tar`, `translate.sh`, `README.txt`) và `dist/hy-mt-vlsp26.tar.gz.md5`.
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
| `translate.sh` báo `service ... is not ready` | Model đang nạp hoặc sai `host:port`; xem log container |

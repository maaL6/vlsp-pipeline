# Hy-MT2 VLSP26 translation service: POST a Markdown document to /translate, get the translation back.
#
# Two targets:
#   full     (default) model baked in, works offline; needs models/hy_mt_merged in the build context
#            (python tools/merge_adapter.py --adapter ADAPTER_DIR --out models/hy_mt_merged)
#   runtime  no weights; mount the merged model at /models/hy_mt_merged
#
#   docker build -t hy-mt-vlsp26 .                      # full image
#   docker run --rm --gpus all -p 8000:8000 hy-mt-vlsp26
#   ./translate.sh localhost:8000 input.md output.md
#
# Folder mode (no service):  docker run --rm --gpus all --entrypoint /app/run.sh \
#   -v /path/in:/input:ro -v /path/out:/output hy-mt-vlsp26 /input /output en-vi
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv python3-pip bash \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH PYTHON=/opt/venv/bin/python MODEL=/models/hy_mt_merged
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt

COPY docmt ./docmt
COPY configs ./configs
COPY tools ./tools
COPY run.sh translate.sh ./
RUN chmod +x run.sh translate.sh

ENV HOST=0.0.0.0 PORT=8000
EXPOSE 8000
ENTRYPOINT ["python", "-m", "docmt.server"]

FROM runtime AS full
ENV HF_HUB_OFFLINE=1
COPY models/hy_mt_merged /models/hy_mt_merged

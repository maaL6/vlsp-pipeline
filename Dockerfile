# Hy-MT2 VLSP26 translation pipeline: folder of .md in, folder of .md out.
#
# Two targets:
#   full     (default) model baked in; needs models/hy_mt_merged in the build context (tools/merge_adapter.py)
#   runtime  no weights; mount the merged model at /models/hy_mt_merged
#
#   docker build -t hy-mt-vlsp26 .                      # full image
#   docker build --target runtime -t hy-mt-vlsp26 .     # without weights
#   docker run --rm --gpus all -v /path/in:/input:ro -v /path/out:/output hy-mt-vlsp26 /input /output en-vi
#   (runtime image: add  -v /path/to/models:/models:ro )
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
COPY run.sh ./run.sh
RUN chmod +x run.sh

ENTRYPOINT ["/app/run.sh"]

FROM runtime AS full
ENV HF_HUB_OFFLINE=1
COPY models/hy_mt_merged /models/hy_mt_merged

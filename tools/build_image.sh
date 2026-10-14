#!/usr/bin/env bash
# Build the complete Docker image from this repo folder alone (needs Docker + NVIDIA Container Toolkit + internet).
#
#   1. put the LoRA adapter in adapter/   (adapter_model.safetensors + adapter_config.json)
#   2. tools/build_image.sh [IMAGE]       default IMAGE: hy-mt-vlsp26
#
# Steps: build the runtime image -> merge adapter + base model inside it (GPU) into models/hy_mt_merged
# -> build the full image with the model baked in. Nothing is installed on the host.
set -euo pipefail
IMAGE="${1:-hy-mt-vlsp26}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f adapter/adapter_model.safetensors ] && [ -f adapter/adapter_config.json ] \
  || { echo "put adapter_model.safetensors and adapter_config.json in $(pwd)/adapter/" >&2; exit 1; }

docker build --target runtime -t "$IMAGE:runtime" .
if [ ! -f models/hy_mt_merged/model.safetensors ]; then
  mkdir -p models
  docker run --rm --gpus all -v "$PWD/adapter:/adapter:ro" -v "$PWD/models:/models" \
    --entrypoint python "$IMAGE:runtime" tools/merge_adapter.py --adapter /adapter --out /models/hy_mt_merged
fi
docker build -t "$IMAGE" .
echo "built $IMAGE (model baked in). Test: docker run --rm --gpus all -p 8000:8000 $IMAGE"

#!/usr/bin/env bash
# Upload the runtime + adapter datasets and push the private-test kernel to Kaggle (T4 x2).
#
#   KAGGLE_API_TOKEN=... OWNER=<kaggle user> ADAPTER_DIR=<adapter dir> kaggle/push.sh
#
# Environment:
#   KAGGLE_API_TOKEN  Kaggle API token (required; never put it in a file)
#   OWNER             Kaggle username that owns the datasets and the kernel (required)
#   PRIVATE_CSV       private test input CSV                                   [dataset/private_test/private_test_input.csv]
#   ADAPTER_DIR       folder with adapter_model.safetensors + adapter_config.json (sha256 30b5fc68...; required)
#   RUN_TAG           suffix of the dataset/kernel names                       [private]
#   KAGGLE            kaggle CLI (>= 1.8, Python >= 3.11)                      [kaggle]
# Output: the kernel writes results_*.zip and predictions_private/{en,vi}/ to /kaggle/working;
# fetch with `kaggle kernels output OWNER/KERNEL -p OUT`, then tools/build_submission.py.
set -euo pipefail
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}" "${OWNER:?set OWNER}" "${ADAPTER_DIR:?set ADAPTER_DIR}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRIVATE_CSV="${PRIVATE_CSV:-$HERE/../dataset/private_test/private_test_input.csv}"
KAGGLE="${KAGGLE:-kaggle}"
TAG="${RUN_TAG:-private}"
RUNTIME_SLUG="vlsp26-runtime-$TAG"
ADAPTER_SLUG="vlsp26-adapter-$TAG"
KERNEL_SLUG="vlsp26-kernel-$TAG"
BUILD="$HERE/build"
rm -rf "$BUILD" && mkdir -p "$BUILD"

echo "==> staging runtime bundle"
python3 "$HERE/build_bundle.py" --csv "$PRIVATE_CSV" --out "$BUILD/runtime" --owner "$OWNER" --slug "$RUNTIME_SLUG"

echo "==> staging adapter"
mkdir -p "$BUILD/adapter"
cp "$ADAPTER_DIR/adapter_model.safetensors" "$ADAPTER_DIR/adapter_config.json" "$BUILD/adapter/"
printf '{"title": "%s", "id": "%s/%s", "licenses": [{"name": "CC0-1.0"}], "isPrivate": true}\n' \
  "$ADAPTER_SLUG" "$OWNER" "$ADAPTER_SLUG" > "$BUILD/adapter/dataset-metadata.json"

echo "==> staging kernel"
mkdir -p "$BUILD/kernel"
cp "$HERE/kernel_private.py" "$BUILD/kernel/kernel_private.py"
cat > "$BUILD/kernel/kernel-metadata.json" <<JSON
{
  "id": "$OWNER/$KERNEL_SLUG",
  "title": "$KERNEL_SLUG",
  "code_file": "kernel_private.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["$OWNER/$ADAPTER_SLUG", "$OWNER/$RUNTIME_SLUG"],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}
JSON

upload() {  # dataset dir, slug, message
  if "$KAGGLE" datasets files "$OWNER/$2" >/dev/null 2>&1; then
    "$KAGGLE" datasets version -p "$1" -m "$3" -r zip -q
  else
    "$KAGGLE" datasets create -p "$1" -r zip -q
  fi
}
echo "==> uploading datasets"
upload "$BUILD/runtime" "$RUNTIME_SLUG" "runtime $TAG"
upload "$BUILD/adapter" "$ADAPTER_SLUG" "adapter $TAG"
for slug in "$RUNTIME_SLUG" "$ADAPTER_SLUG"; do
  for _ in $(seq 60); do
    [ "$("$KAGGLE" datasets status "$OWNER/$slug" 2>/dev/null || true)" = "ready" ] && break
    sleep 5
  done
done

echo "==> pushing kernel $OWNER/$KERNEL_SLUG"
"$KAGGLE" kernels push -p "$BUILD/kernel"
"$KAGGLE" kernels status "$OWNER/$KERNEL_SLUG"

#!/usr/bin/env bash
# Pack the Docker image and the client script into one archive for submission.
#
#   tools/package_docker.sh [IMAGE] [OUT_DIR]          defaults: hy-mt-vlsp26  dist
#
# Writes OUT_DIR/hy-mt-vlsp26.tar.gz (the archive to upload) and OUT_DIR/hy-mt-vlsp26.tar.gz.md5.
# The archive holds: image.tar (docker save), translate.sh and README.md (a copy of USAGE.md).
set -euo pipefail
IMAGE="${1:-hy-mt-vlsp26}"; OUT="${2:-dist}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$OUT"
docker image inspect "$IMAGE" >/dev/null
docker save "$IMAGE" -o "$STAGE/image.tar"
cp "$HERE/translate.sh" "$STAGE/translate.sh"
cp "$HERE/USAGE.md" "$STAGE/README.md"
tar -C "$STAGE" -czf "$OUT/hy-mt-vlsp26.tar.gz" image.tar translate.sh README.md
if command -v md5sum >/dev/null; then md5sum "$OUT/hy-mt-vlsp26.tar.gz" > "$OUT/hy-mt-vlsp26.tar.gz.md5"
else md5 -r "$OUT/hy-mt-vlsp26.tar.gz" > "$OUT/hy-mt-vlsp26.tar.gz.md5"; fi
cat "$OUT/hy-mt-vlsp26.tar.gz.md5"

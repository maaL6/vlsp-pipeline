#!/usr/bin/env bash
# Pack the Docker image and the client script into one archive for submission.
#
#   tools/package_docker.sh [IMAGE] [OUT_DIR]          defaults: hy-mt-vlsp26  dist
#
# Writes OUT_DIR/hy-mt-vlsp26.tar.gz (the archive to upload) and OUT_DIR/hy-mt-vlsp26.tar.gz.md5.
# The archive holds: image.tar (docker save), translate.sh and README.txt.
set -euo pipefail
IMAGE="${1:-hy-mt-vlsp26}"; OUT="${2:-dist}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$OUT"
docker image inspect "$IMAGE" >/dev/null
docker save "$IMAGE" -o "$STAGE/image.tar"
cp "$HERE/translate.sh" "$STAGE/translate.sh"
cat > "$STAGE/README.txt" <<TXT
Hy-MT2 VLSP26 document-level legal MT (EN<->VI), self-contained and offline.

  docker load -i image.tar
  docker run --rm --gpus all -p 8000:8000 $IMAGE        # model loads at start; GET /health is 200 when ready
  ./translate.sh HOST:PORT INPUT.md OUTPUT.md            # one document, e.g. ./translate.sh localhost:8000 in.md out.md
  ./translate.sh HOST:PORT private_test.csv results.csv  # private test CSV -> submission CSV (sample_id,direction,translation)
  # or without HTTP: docker run --rm --gpus all -v IN_DIR:/in:ro -v OUT_DIR:/out $IMAGE --csv /in/private_test.csv --out /out/results.csv

translate.sh posts the document to http://HOST:PORT/translate, detects the direction from the text,
writes the UTF-8 Markdown translation and prints statistics (seconds, words/second, peak GPU memory).
TXT
tar -C "$STAGE" -czf "$OUT/hy-mt-vlsp26.tar.gz" image.tar translate.sh README.txt
if command -v md5sum >/dev/null; then md5sum "$OUT/hy-mt-vlsp26.tar.gz" > "$OUT/hy-mt-vlsp26.tar.gz.md5"
else md5 -r "$OUT/hy-mt-vlsp26.tar.gz" > "$OUT/hy-mt-vlsp26.tar.gz.md5"; fi
cat "$OUT/hy-mt-vlsp26.tar.gz.md5"

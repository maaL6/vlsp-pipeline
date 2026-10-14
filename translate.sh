#!/usr/bin/env bash
# Send one Markdown document to the running translation service and write the translation.
#
#   ./translate.sh HOST:PORT INPUT OUTPUT          e.g. ./translate.sh localhost:8000 in.md out.md
#
# INPUT.md  -> OUTPUT.md : one Markdown document, direction detected automatically.
# INPUT.csv -> OUTPUT.csv: the private test CSV (sample_id, category, source_language, source_markdown)
#                          -> the submission CSV (sample_id, direction, translation).
# Waits up to WAIT_SECONDS [1800] for the service to finish loading the model.
# Prints statistics (direction, words, seconds, words/second, peak GPU memory) to stdout.
set -euo pipefail
if [ $# -ne 3 ]; then sed -n '2,10p' "$0"; exit 2; fi
ADDR="$1"; IN="$2"; OUT="$3"
WAIT="${WAIT_SECONDS:-1800}"
[ -f "$IN" ] || { echo "input not found: $IN" >&2; exit 2; }

for ((i = 0; i < WAIT; i += 2)); do
  curl -s -f --noproxy "*" -o /dev/null "http://$ADDR/health" && break
  sleep 2
done
curl -s -f --noproxy "*" -o /dev/null "http://$ADDR/health" || { echo "service at $ADDR is not ready" >&2; exit 1; }

HEADERS="$(mktemp)"; trap 'rm -f "$HEADERS"' EXIT
mkdir -p "$(dirname "$OUT")"
START=$SECONDS
ENDPOINT=translate; case "$IN" in *.csv|*.CSV) ENDPOINT=translate_csv ;; esac
CODE="$(curl -s -S --noproxy "*" -o "$OUT" -D "$HEADERS" -w '%{http_code}' -X POST \
  -H 'Content-Type: text/markdown; charset=utf-8' --data-binary @"$IN" "http://$ADDR/$ENDPOINT")"
if [ "$CODE" != "200" ]; then
  echo "translation failed (HTTP $CODE): $(cat "$OUT")" >&2
  rm -f "$OUT"
  exit 1
fi
if [ "$ENDPOINT" = translate_csv ]; then
  echo "rows=$(python3 -c 'import csv,sys; csv.field_size_limit(10**9); print(sum(1 for _ in csv.reader(open(sys.argv[1], encoding="utf-8-sig", newline=""))) - 1)' "$OUT" 2>/dev/null || echo n/a) wall_seconds=$((SECONDS - START))"
  exit 0
fi
header() { grep -i "^$1:" "$HEADERS" | head -1 | sed -E 's/^[^:]*:[[:space:]]*//' | tr -d '\r'; }
PEAK="$(header X-Peak-Vram-Bytes)"
echo "direction=$(header X-Direction) status=$(header X-Status) input_words=$(header X-Input-Words)" \
     "seconds=$(header X-Seconds) words_per_second=$(header X-Words-Per-Second)" \
     "peak_gpu_memory_mb=$([ -n "$PEAK" ] && echo $((PEAK / 1048576)) || echo n/a) wall_seconds=$((SECONDS - START))"

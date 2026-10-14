#!/usr/bin/env bash
# Final contract: translate every .md under INPUT_DIR into OUTPUT_DIR (same relative tree).
#
#   ./run.sh INPUT_DIR OUTPUT_DIR en-vi|vi-en
#
# Environment (defaults in brackets):
#   MODEL      HF model dir of the merged model (model.safetensors + tokenizer)    [./models/hy_mt_merged]
#   TOKENIZER  tokenizer.json                                                [$MODEL/tokenizer.json]
#   CONFIG     preset JSON                                                   [configs/hy_mt_vlsp26.json]
#              (batch 8, fp16 greedy, legal prose blocks)
#   RUN_ID     run id written to logs                                        [final_<direction>_<UTC time>]
#   WORK_DIR   logs/traces/cache (not part of the output)                    [OUTPUT_DIR.work]
#   PYTHON     interpreter with torch+CUDA, transformers>=5.6, tokenizers    [python3]
#   BACKEND    hf (GPU) | mock (install check only: echoes the source, NOT a translation)  [hf]
# Exit code != 0 when a document failed or a prediction is missing (see WORK_DIR/logs).
set -euo pipefail
if [ $# -ne 3 ]; then sed -n '2,15p' "$0"; exit 2; fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IN="$1"; OUT="$2"; DIR="$3"
MODEL="${MODEL:-$HERE/models/hy_mt_merged}"
TOKENIZER="${TOKENIZER:-$MODEL/tokenizer.json}"
CONFIG="${CONFIG:-$HERE/configs/hy_mt_vlsp26.json}"
RUN_ID="${RUN_ID:-final_${DIR}_$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON="${PYTHON:-python3}"
BACKEND="${BACKEND:-hf}"
ARGS=(run-dir --input-dir "$IN" --output-dir "$OUT" --direction "$DIR" --config "$CONFIG"
      --run-id "$RUN_ID" --backend "$BACKEND" --model "$MODEL" --tokenizer "$TOKENIZER")
if [ -n "${WORK_DIR:-}" ]; then ARGS+=(--work-dir "$WORK_DIR"); fi
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" PYTHONIOENCODING=utf-8 exec "$PYTHON" -m docmt "${ARGS[@]}"

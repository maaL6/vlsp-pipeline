#!/usr/bin/env bash
# Download kernel outputs and score the variants side by side.
#
#   ./fetch_and_compare.sh [OUT_DIR]        (default: ./results)
#
# KERNELS lists "owner:variant:token_env" entries; each token is read from the named environment
# variable (default: KAGGLE_API_TOKEN for all). Default:
#   KERNELS="wann179:markers wann179:nomarker maal06:newline:KAGGLE_TOKEN_MAAL06"
# Needs sacrebleu (pip install -r ../../../benchmark/requirements.txt).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/../../../benchmark" && pwd)"
KAGGLE="${KAGGLE:-kaggle}"
KERNELS="${KERNELS:-wann179:markers wann179:nomarker maal06:newline:KAGGLE_TOKEN_MAAL06}"
PYTHON="${PYTHON:-python3}"
OUT="${1:-$HERE/results}"
HYPS=()
for k in $KERNELS; do
  IFS=: read -r owner v tok_env <<< "$k"
  tok_env="${tok_env:-KAGGLE_API_TOKEN}"; tok="${!tok_env:-}"
  d="$OUT/$v"; rm -rf "$d"; mkdir -p "$d"
  KAGGLE_API_TOKEN="$tok" "$KAGGLE" kernels output "$owner/vlsp26-lam-pt50-$v" -p "$d" \
    --file-pattern "results_$v.zip|kernel_status.json" -q
  ( cd "$d" && unzip -q -o "results_$v.zip" )
  echo "== $v"; "$PYTHON" -c "import json,sys; s=json.load(open(sys.argv[1])); print(json.dumps(s['directions'], indent=1))" "$d/kernel_status.json"
  HYPS+=("$d/hyp_$v")
done
"$PYTHON" "$BENCH/eval_markdown.py" --direction both --allow-missing --hyp "${HYPS[@]}" --save "$OUT/scores_compare.json" \
  | tee "$OUT/scores_compare.txt"

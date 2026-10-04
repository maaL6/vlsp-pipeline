#!/usr/bin/env bash
# Upload pipeline-v1 + benchmark/ as a private Kaggle dataset, then push one private GPU kernel per
# variant (markers = A, newline = B chosen, nomarker = C). Each kernel runs the full pipeline on the 50-doc public test in both
# directions and scores it with benchmark/eval_markdown.py.
#
#   KAGGLE_API_TOKEN=... ./push.sh [markers|nomarker|newline|all]   (default: all = markers + nomarker)
#
# Environment (defaults in brackets):
#   KAGGLE        kaggle CLI (>= 1.8, reads KAGGLE_API_TOKEN)               [kaggle]
#   OWNER         Kaggle user                                               [wann179]
#   MODEL_DS      dataset with the A1 merged model                          [$OWNER/vlsp26-hy-a1-merged]
#   BUNDLE_DS     dataset slug for code + public test                       [vlsp26-lam-pipeline-v1-pt50]
#   RESUME        1 = also mount the kernel's own previous output (continue after a timeout)  [0]
# The token is read from the environment only; it is never written to disk by this script.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="$(cd "$HERE/../.." && pwd)"
VLSP="$(cd "$PIPE/.." && pwd)"
KAGGLE="${KAGGLE:-kaggle}"
OWNER="${OWNER:-wann179}"
MODEL_DS="${MODEL_DS:-$OWNER/vlsp26-hy-a1-merged}"
BUNDLE_DS="${BUNDLE_DS:-vlsp26-lam-pipeline-v1-pt50}"
RESUME="${RESUME:-0}"
WHICH="${1:-all}"
BUILD="$HERE/build"
: "${KAGGLE_API_TOKEN:?set KAGGLE_API_TOKEN}"

# -- 1. bundle dataset -------------------------------------------------------------------------
rm -rf "$BUILD/bundle"; mkdir -p "$BUILD/bundle/pipeline-v1" "$BUILD/bundle/benchmark/data"
( cd "$PIPE" && cp -R docmt configs run.sh requirements.txt "$BUILD/bundle/pipeline-v1/" )
( cd "$VLSP/benchmark" && cp -R common.py eval_markdown.py export_public_test.py requirements.txt normalization \
     "$BUILD/bundle/benchmark/" && cp data/en-vi-public_test.csv "$BUILD/bundle/benchmark/data/" )
find "$BUILD/bundle" \( -name __pycache__ -o -name '*.pyc' -o -name .DS_Store \) -prune -exec rm -rf {} +
GIT_HEAD="$(git -C "$PIPE" rev-parse --short HEAD)$(git -C "$PIPE" diff --quiet HEAD -- docmt configs || echo -dirty)"
python3 - "$BUILD/bundle" "$GIT_HEAD" <<'EOF'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(root.rglob("*")) if p.is_file()}
(root / "pipeline_bundle_manifest.json").write_text(json.dumps({"git_head": sys.argv[2], "files": files}, indent=1))
print(f"bundle: {len(files)} files, pipeline-v1 {sys.argv[2]}")
EOF
cat > "$BUILD/bundle/dataset-metadata.json" <<EOF
{"title": "$BUNDLE_DS", "id": "$OWNER/$BUNDLE_DS", "licenses": [{"name": "other"}]}
EOF
if "$KAGGLE" datasets files "$OWNER/$BUNDLE_DS" >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$BUILD/bundle" -m "pipeline-v1 $GIT_HEAD" -r zip -q
else
  "$KAGGLE" datasets create -p "$BUILD/bundle" -r zip -q    # private by default
fi
echo "waiting for dataset $OWNER/$BUNDLE_DS to be ready..."
for _ in $(seq 60); do
  s="$("$KAGGLE" datasets status "$OWNER/$BUNDLE_DS" 2>/dev/null || true)"
  [ "$s" = "ready" ] && break; sleep 10
done
echo "dataset status: ${s:-unknown}"

# -- 2. one kernel per variant -------------------------------------------------------------------
push_kernel() {
  local v="$1" slug="vlsp26-lam-pt50-$1" dir="$BUILD/kernel_$1"
  rm -rf "$dir"; mkdir -p "$dir"
  sed "s/__VARIANT__/$v/" "$HERE/kernel_public_test.py" > "$dir/$slug.py"
  local kernel_sources="[]"
  [ "$RESUME" = "1" ] && kernel_sources="[\"$OWNER/$slug\"]"
  cat > "$dir/kernel-metadata.json" <<EOF
{
  "id": "$OWNER/$slug",
  "title": "$slug",
  "code_file": "$slug.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["$MODEL_DS", "$OWNER/$BUNDLE_DS"],
  "kernel_sources": $kernel_sources,
  "competition_sources": [],
  "model_sources": [],
  "machine_shape": "NvidiaTeslaT4"
}
EOF
  "$KAGGLE" kernels push -p "$dir"
  echo "pushed $OWNER/$slug  (status: $KAGGLE kernels status $OWNER/$slug)"
}
case "$WHICH" in
  all) push_kernel markers; push_kernel nomarker ;;
  markers|nomarker|newline) push_kernel "$WHICH" ;;
  *) echo "usage: $0 [markers|nomarker|newline|all]"; exit 2 ;;
esac

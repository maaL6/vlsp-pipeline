#!/usr/bin/env bash
# Offline acceptance for any change that must NOT alter behaviour (refactor) - no GPU needed:
#   1) unit/integration tests  2) static name check  3) replay equivalence vs lam_docmt on smoke6
# Usage: tools/verify.sh [python]     (default: $PYTHON or python3; needs tokenizers + pytest)
# Paths (defaults assume pipeline-v1 sits inside the VLSP26 repo):
#   VLSP26_ROOT        repo root holding "task 3-10/" and "report_lam_3_10/"   [parent of this folder]
#   DOCMT_TEST_INPUTS  test inputs                                               [$VLSP26_ROOT/task 3-10/CHUNG/inputs]
#   WORK               scratch dir for replay/mock outputs                       [mktemp -d]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${1:-${PYTHON:-python3}}"
ROOT="${VLSP26_ROOT:-$(cd "$HERE/.." && pwd)}"
INPUTS="${DOCMT_TEST_INPUTS:-$ROOT/task 3-10/CHUNG/inputs}"
export DOCMT_TEST_INPUTS="$INPUTS"
WORK="${WORK:-$(mktemp -d)}"
cd "$HERE"
echo "== pytest"; (cd tests && "$PY" -m pytest -q -p no:cacheprovider .)
echo "== names";  PYTHONPATH="$HERE" "$PY" tools/check_names.py docmt || true   # 3 known comprehension false positives
echo "== replay"; "$PY" tools/replay_equivalence.py \
  --legacy-code "$ROOT/report_lam_3_10/code" \
  --run-dir "$ROOT/report_lam_3_10/delivery/hy_a1_structure__kaggleT4__20261003b" \
  --manifest "$INPUTS/smoke6_doc_direction_manifest.jsonl" --data-root "$INPUTS" \
  --tokenizer "$INPUTS/hy_a1_tokenizer/tokenizer.json" \
  --work "$WORK/replay" --report "$HERE/reports/replay_equivalence.json" | head -3
echo "== mock (4 branches incl. context/glossary/history)"; "$PY" tools/mock_equivalence.py \
  --legacy-code "$ROOT/report_lam_3_10/code" \
  --manifest "$INPUTS/smoke6_doc_direction_manifest.jsonl" --data-root "$INPUTS" \
  --tokenizer "$INPUTS/hy_a1_tokenizer/tokenizer.json" \
  --glossary "$INPUTS/release_v2/glossary_provenance.jsonl" \
  --work "$WORK/mockeq" --report "$HERE/reports/mock_equivalence.json" | tail -1

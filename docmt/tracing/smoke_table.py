from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from ..core.hashing import read_jsonl
from .writer import LOG_FILES, SMOKE_COLUMNS

# -- smoke table -----------------------------------------------------------------------------------
MARKER_EOS_CODES = ("marker_", "unassigned_text", "segment_empty", "empty_output", "no_eos",
                    "internal_marker_leak", "operational_error", "prompt_too_long")

TABLE_NUMBER_HARD = ("protected_heading_lost", "table_cell_multiline")

TABLE_NUMBER_WARN = ("number_mismatch", "label_changed")

def _code(c: str) -> str:
    return str(c).split(":", 1)[0]

def _summ(codes) -> str:
    cnt = Counter(_code(c) for c in codes)
    return "; ".join(f"{k}x{v}" if v > 1 else k for k, v in sorted(cnt.items()))

def smoke_row(item: dict, status: dict | None, n_trace: int, run_id: str) -> dict:
    """One SMOKE_RESULT.csv row from a document_status record (None -> not run)."""
    row = {k: "" for k in SMOKE_COLUMNS}
    row.update(sample_id=item["sample_id"], direction=item["direction"], run_id=run_id,
               coverage_status="not_checked", marker_eos_status="not_checked",
               table_number_status="not_checked", name_status="not_checked")
    if status is None:
        row["notes"] = "not run in this run directory"
        row["review_required"] = "yes"
        return row
    hard = list(status.get("selected_hard_failures") or [])
    doc_hard = list(status.get("doc_hard_failures") or [])
    warns = list(status.get("selected_warnings") or [])
    st = status.get("status")
    row["output_path"] = status.get("output_path") or ""
    rt = status.get("roundtrip") or {}
    row["source_span"] = (f"chars 0-{status.get('source_chars', '?')}; nodes {status.get('nodes', '?')}; "
                          f"roundtrip {'ok' if rt.get('ok') else 'FAIL'}")
    row["retry_count"] = str(status.get("recovery_units", 0))
    row["evidence"] = (f"logs/document_status.jsonl; logs/generation_trace.jsonl ({n_trace} requests)")
    notes = [f"status={st}"]
    if st == "failed":
        row["coverage_status"] = "fail"
        notes.append(f"error={status.get('error')}")
    else:
        cov_ok = (rt.get("ok") and not status.get("uncovered_nodes") and not status.get("unresolved_empty")
                  and (status.get("doc_check") or {}).get("coverage_ok", True) is not False)
        row["coverage_status"] = "pass" if cov_ok else "fail"
        all_hard = hard + doc_hard
        if not status.get("validator_available", True):
            row["marker_eos_status"] = "not_checked"
        else:
            bad = [c for c in all_hard if any(_code(c).startswith(p) for p in MARKER_EOS_CODES)]
            row["marker_eos_status"] = "fail" if bad else "pass"
        if not status.get("validator_available", True):
            row["table_number_status"] = "not_checked"
        elif any(_code(c) in TABLE_NUMBER_HARD or "heading" in _code(c) for c in all_hard):
            row["table_number_status"] = "fail"
        elif any(_code(w) in TABLE_NUMBER_WARN for w in warns) or status.get("broken_table_nodes"):
            row["table_number_status"] = "needs_review"
        else:
            row["table_number_status"] = "pass"
        if status.get("signature_name_nodes"):
            row["name_status"] = "pass"
            notes.append(f"{status['signature_name_nodes']} signer-name node(s) copied verbatim; other names not checked")
        elif status.get("signature_nodes"):
            row["name_status"] = "needs_review"
            notes.append("signature block without a confidently detected name")
        if status.get("unresolved_units"):
            notes.append(f"unresolved_units={len(status['unresolved_units'])}")
        if status.get("unresolved_empty"):
            notes.append(f"unresolved_empty={len(status['unresolved_empty'])}")
        if status.get("recovered_units"):
            notes.append(f"recovered_units={status['recovered_units']}")
    if status.get("op_retries") or status.get("op_errors"):
        notes.append(f"op_retries={status.get('op_retries', 0)} op_errors={status.get('op_errors', 0)}")
    if status.get("backend") == "mock":
        notes.append("MOCK backend output (echo, not a translation)")
    notes.append("meaning not reviewed (automatic checks only)")
    row["hard_failures"] = "; ".join(sorted(set(hard + doc_hard)))
    row["warnings"] = _summ(warns)
    flagged = any(row[c] in ("fail", "needs_review", "not_checked") for c in
                  ("coverage_status", "marker_eos_status", "table_number_status", "name_status"))
    row["review_required"] = "yes" if (flagged or warns or hard or doc_hard or st != "completed") else "no"
    row["notes"] = "; ".join(notes)
    return row

def write_smoke_table(run_dir, items, out_path=None, run_id: str | None = None) -> Path:
    """tests/smoke_result.csv from the run logs.
    `items`: [{"sample_id", "direction"}] in the order the table should list them."""
    run_dir = Path(run_dir)
    statuses = {(r.get("doc_id"), r.get("direction")): r for r in read_jsonl(run_dir / "logs" / LOG_FILES["status"])}
    traces = Counter((r.get("doc_id"), r.get("direction")) for r in read_jsonl(run_dir / "logs" / LOG_FILES["trace"]))
    rid = run_id or next((r.get("run_id") for r in statuses.values() if r.get("run_id")), run_dir.name)
    out = Path(out_path) if out_path else run_dir / "tests" / "smoke_result.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(SMOKE_COLUMNS), quoting=csv.QUOTE_ALL)
        w.writeheader()
        for it in items:
            key = (it["sample_id"], it["direction"])
            w.writerow(smoke_row(it, statuses.get(key), traces.get(key, 0), rid))
    return out

# Verification scripts

These CPU-only scripts inspect saved outputs; they do not train models or run
translation inference. Run them from the repository root.

- `reproduce_statistics.py`: rescoring and document bootstrap. Requires
  `sacrebleu==2.6.0` and NumPy. Pass `--csv`, `--abc-root`, `--final-pred`, and
  `--output`; `--help` lists the seed and resampling options. The grouping root
  must contain `A_gom_node_marker/predictions`,
  `B_gom_node_khong_marker/predictions`, and `C_khong_gom_khong_marker/predictions`.
- `audit_unresolved.py`: classify final structural failures and verify saved
  prediction hashes against document diagnostics. Uses the Python standard
  library. Pass `--delivery-root`, `--pred`, and `--output`. The delivery root
  must contain `work_public_en-vi/logs/document_status.jsonl` and its `vi-en`
  counterpart; prediction roots contain `vi/` and `en/` subdirectories.

Saved run logs and translations are external inputs, not bundled here. Use
`--output paper/acl/verification/statistics.json` or `unresolved.json` as
appropriate. Generated JSON reports and internal editorial audits remain local
and are excluded from Git. Neither script verifies the provenance of the
original main-table baseline/Index scores.

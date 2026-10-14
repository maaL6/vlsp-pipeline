"""Command-line entry points (python -m docmt <command>).

  run               manifest + data root -> delivery layout (dev/smoke runs)
  run-dir           folder of .md -> folder of .md with the same tree (final/Docker contract)
  inspect           dump intermediate results of steps s01..s05 without a model (--stop-after)
  parse-audit       parser roundtrip report
  validate-existing source-only diagnostics of existing predictions
  smoke-table       smoke_result.csv of a run directory
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .core.config import RunConfig, make_config
from .core.tok import HyTok


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _overrides(parser, pairs) -> dict:
    out = {}
    for value in pairs:
        key, sep, val = value.partition("=")
        if not sep:
            parser.error("--set requires KEY=VALUE")
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        out[key] = val
    return out


def build_config(parser, args) -> RunConfig:
    """Preset JSON (--config) or branch preset, then explicit flags, then --set overrides."""
    given = dict(run_id=args.run_id, backend=args.backend, tokenizer_path=args.tokenizer, model=args.model)
    if args.config:
        data = json.loads(Path(args.config).read_text(encoding="utf-8"))
        data.pop("config_sha256", None)
        cfg = RunConfig().apply(data)
        cfg.apply({k: v for k, v in given.items() if v not in (None, "")})
    else:
        cfg = make_config(args.branch, **{k: v for k, v in given.items() if v is not None})
    cfg.apply(_overrides(parser, args.set))
    if cfg.reference_available_at_inference or cfg.normalize_reference:
        parser.error("inference must not use or normalize references")
    return cfg


def folder_rows(input_dir: Path, direction: str) -> list:
    """One row per .md file under input_dir (sorted); sample_id = relative path without .md."""
    rows = []
    for p in sorted(input_dir.rglob("*.md")):
        rel = p.relative_to(input_dir)
        sid = rel.with_suffix("").as_posix()
        rows.append({"sample_id": sid, "category": rel.parts[0] if len(rel.parts) > 1 else "",
                     "direction": direction, "source_path": rel.as_posix()})
    return rows


def _add_run_args(p, need_out=True):
    p.add_argument("--config", help="preset JSON (configs/*.json); flags below override it")
    p.add_argument("--branch", default="hy_mt_vlsp26")
    p.add_argument("--run-id", default=None)
    p.add_argument("--backend", choices=("hf", "mock"), default=None)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--directions", default="en-vi,vi-en")
    run.add_argument("--limit", type=int)
    run.add_argument("--dry-run", action="store_true")
    _add_run_args(run)
    rdir = sub.add_parser("run-dir")
    rdir.add_argument("--input-dir", required=True)
    rdir.add_argument("--output-dir", required=True)
    rdir.add_argument("--direction", required=True, choices=("en-vi", "vi-en"))
    rdir.add_argument("--work-dir", help="logs/traces (default: <output-dir>.work)")
    _add_run_args(rdir)
    insp = sub.add_parser("inspect")
    insp.add_argument("--manifest")
    insp.add_argument("--data-root")
    insp.add_argument("--input-dir", help="folder mode (with --direction)")
    insp.add_argument("--direction", choices=("en-vi", "vi-en"))
    insp.add_argument("--directions", default="en-vi,vi-en")
    insp.add_argument("--stop-after", required=True, choices=("parse", "payload", "split", "pack", "prompt"))
    insp.add_argument("--out", required=True, help="output JSONL")
    insp.add_argument("--limit", type=int)
    _add_run_args(insp)
    audit = sub.add_parser("parse-audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--tokenizer", required=True)
    audit.add_argument("--out", required=True)
    audit.add_argument("--both-directions", action="store_true")
    check = sub.add_parser("validate-existing")
    check.add_argument("--manifest", required=True)
    check.add_argument("--data-root", required=True)
    check.add_argument("--pred-root", required=True)
    check.add_argument("--out", required=True)
    check.add_argument("--label", default="existing")
    table = sub.add_parser("smoke-table")
    table.add_argument("--run-dir", required=True)
    table.add_argument("--manifest", required=True)
    table.add_argument("--data-root", required=True)
    args = parser.parse_args(argv)

    if args.command in ("run", "run-dir"):
        from .backends import make_backend
        from .engine.api import run_manifest
        cfg = build_config(parser, args)
        if not cfg.run_id or cfg.run_id == "TO_FILL":
            parser.error("--run-id is required (in the preset or on the command line)")
        tok = HyTok(args.tokenizer)
        if args.command == "run":
            rows, data_root, out, dirs = read_rows(args.manifest), args.data_root, args.out, args.directions
            manifest_path, limit, dry = args.manifest, args.limit, args.dry_run
        else:
            in_dir = Path(args.input_dir)
            rows, data_root, dirs = folder_rows(in_dir, args.direction), in_dir, args.direction
            out = Path(args.work_dir) if args.work_dir else Path(str(Path(args.output_dir)) + ".work")
            manifest_path, limit, dry = None, None, False
        backend = None if dry else make_backend(cfg.backend, cfg, tok)
        result = run_manifest(rows, data_root, out, cfg, backend, tok, dirs, manifest_path=manifest_path,
                              limit=limit, dry_run=dry, progress=lambda m: print(m, flush=True))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        failed = int(any(v.get("statuses", {}).get("failed", 0) for v in result.values())) if not dry else 0
        if args.command == "run-dir":
            # final layout: same relative tree as the input, predictions only (logs stay in work dir)
            missing = []
            dst_root = Path(args.output_dir)
            for r in rows:
                src = Path(out) / "predictions" / args.direction / (r["sample_id"] + ".md")
                dst = dst_root / (r["sample_id"] + ".md")
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(src, dst)
                else:
                    missing.append(r["sample_id"])
            print(json.dumps({"output_dir": str(dst_root), "written": len(rows) - len(missing),
                              "missing": missing, "work_dir": str(out)}, ensure_ascii=False, indent=2))
            failed = failed or int(bool(missing))
        return failed
    if args.command == "inspect":
        from .engine.inspect import inspect_items
        from .inputs.sources import expand_rows
        cfg = build_config(parser, args)
        if args.input_dir:
            if not args.direction:
                parser.error("--input-dir needs --direction")
            items = expand_rows(folder_rows(Path(args.input_dir), args.direction), args.input_dir, [args.direction])
        else:
            if not (args.manifest and args.data_root):
                parser.error("inspect needs --manifest/--data-root or --input-dir/--direction")
            items = expand_rows(read_rows(args.manifest), args.data_root, args.directions)
        if args.limit:
            items = items[: args.limit]
        stats = inspect_items(items, cfg, HyTok(args.tokenizer), args.stop_after, Path(args.out))
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    rows = read_rows(args.manifest)
    if args.command == "parse-audit":
        from .steps.s01_parse_audit import parse_audit
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result = parse_audit(rows, args.data_root, HyTok(args.tokenizer), args.out, args.both_directions)
        print(json.dumps(result["totals"], ensure_ascii=False, indent=2))
        return int(bool(result["totals"]["failed"]))
    if args.command == "smoke-table":
        from .inputs.sources import expand_rows
        from .tracing.smoke_table import write_smoke_table
        items = expand_rows(rows, args.data_root)
        write_smoke_table(Path(args.run_dir), [dict(sample_id=x.sample_id, direction=x.direction) for x in items])
        return 0
    from .inputs.sources import expand_rows
    from .steps.s10_doc_check import audit_plain_output
    documents = []
    for item in expand_rows(rows, args.data_root):
        prediction = Path(args.pred_root) / item.direction / (item.sample_id + ".md")
        record = dict(sample_id=item.sample_id, direction=item.direction, prediction=str(prediction))
        if not prediction.exists() or not prediction.stat().st_size:
            record.update(status="failed", error="missing_or_empty_prediction")
        else:
            record.update(status="completed", diagnostics=audit_plain_output(
                item.source_path.read_bytes().decode("utf-8"), prediction.read_bytes().decode("utf-8"),
                item.src_lang, item.tgt_lang))
        documents.append(record)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(label=args.label, documents=documents,
                                     meaning_review="not_checked"), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(documents)} predictions checked; diagnostics: {output}")
    return int(any(x["status"] == "failed" for x in documents))


if __name__ == "__main__":
    raise SystemExit(main())

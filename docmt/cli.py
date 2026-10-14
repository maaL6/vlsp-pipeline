"""Command-line entry point: python -m docmt run-dir

  run-dir   folder of .md -> folder of .md with the same tree (final/Docker contract)
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .core.config import RunConfig, make_config
from .core.tok import HyTok


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
    if cfg.source_context or cfg.soft_glossary or cfg.target_history:
        parser.error("source_context, soft_glossary and target_history are not supported")
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
    p.add_argument("--backend", choices=("hf",), default=None)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rdir = sub.add_parser("run-dir")
    rdir.add_argument("--input-dir", required=True)
    rdir.add_argument("--output-dir", required=True)
    rdir.add_argument("--direction", required=True, choices=("en-vi", "vi-en"))
    rdir.add_argument("--work-dir", help="logs/traces (default: <output-dir>.work)")
    _add_run_args(rdir)
    args = parser.parse_args(argv)

    from .backends import make_backend
    from .engine.api import run_manifest
    cfg = build_config(parser, args)
    if not cfg.run_id or cfg.run_id == "TO_FILL":
        parser.error("--run-id is required (in the preset or on the command line)")
    tok = HyTok(args.tokenizer)
    in_dir = Path(args.input_dir)
    rows = folder_rows(in_dir, args.direction)
    out = Path(args.work_dir) if args.work_dir else Path(str(Path(args.output_dir)) + ".work")
    backend = make_backend(cfg.backend, cfg, tok)
    result = run_manifest(rows, in_dir, out, cfg, backend, tok, args.direction, manifest_path=None,
                          limit=None, dry_run=False, progress=lambda m: print(m, flush=True))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    failed = int(any(v.get("statuses", {}).get("failed", 0) for v in result.values()))
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
    return failed or int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())

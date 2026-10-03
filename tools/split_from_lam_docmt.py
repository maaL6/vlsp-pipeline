"""Mechanical split of report_lam_3_10/code/lam_docmt into pipeline-v1/docmt (one-off, kept for provenance).

Every top-level statement (function, class, constant, try-block) of the source package is copied
verbatim into exactly one target module given by PLAN. Intra-package imports are regenerated from
the names each target uses; stdlib/third-party imports are copied when used. Behaviour is not
changed here; manual edits after the split are listed in pipeline-v1/CHANGE_NOTES.md.

  python tools/split_from_lam_docmt.py SRC_PKG_DIR OUT_PKG_DIR
"""
from __future__ import annotations

import ast
import sys
from collections import defaultdict
from pathlib import Path

# source module -> (default target, {name: target})
PLAN = {
    "types": ("core.datatypes", {}),
    "config": ("core.config", {}),
    "tok": ("core.tok", {}),
    "prompts": ("steps.s04_prompt", {}),
    "parse": ("steps.s01_parse_lines", {
        "unwrap": "steps.s02_payload", "make_payload": "steps.s02_payload",
        "_Parser": "steps.s01_parse", "_spans_to_nodes": "steps.s01_parse", "_fallback_nodes": "steps.s01_parse",
        "check_partition": "steps.s01_parse", "parse_document": "steps.s01_parse", "reconstruct": "steps.s01_parse",
        "PARSER_VERSION": "steps.s01_parse", "TABLE_CAP": "steps.s01_parse_lines",
        "_sha256": "steps.s01_parse_audit", "audit_document": "steps.s01_parse_audit",
        "_row_items": "steps.s01_parse_audit", "parse_audit": "steps.s01_parse_audit",
    }),
    "units": ("steps.s03_split", {
        "__all__": None,
        "MARKER_RE": "steps.s03_pack", "_NS_FULL": "steps.s03_pack", "MAX_MARKER_INDEX": "steps.s03_pack",
        "_sha": "steps.s03_pack",
        "marker_namespace": "steps.s03_pack", "marker": "steps.s03_pack", "_ns_of": "steps.s03_pack",
        "_index_of": "steps.s03_pack", "render_payload": "steps.s03_pack", "_greedy": "steps.s03_pack", "pack_units": "steps.s03_pack",
        "target_budget": "steps.s05_budget", "output_budget": "steps.s05_budget", "budget_fits": "steps.s05_budget",
        "_near_miss_re": "steps.s06_parse_output", "parse_output": "steps.s06_parse_output",
        "merge_child_segments": "steps.s06_parse_output",
        "_span_text": "steps.s08_recovery", "_halves": "steps.s08_recovery", "recovery_units": "steps.s08_recovery",
    }),
    "validate": ("steps.s07_validate_unit", {
        **{n: "checks.common" for n in ("FFFD", "ANY_MARKER", "_LETTER", "_ALNUM", "_nfc", "_ws", "_dedupe")},
        **{n: "checks.numbers" for n in (
            "BIG_DIGITS", "_NUM", "_LABEL_START", "_LABEL_MID", "_MONTHS", "_MONTH_ALWAYS", "_MONTH_RE",
            "_HEAD_ROMAN_NUM", "NumItem", "_canon_int", "_canon_dec", "_parse_token", "_roman_value",
            "_map_months", "_roman_heading_sub", "number_items", "number_tokens", "_is_big", "display_value",
            "compare_numbers", "_vkey", "_contexts")},
        **{n: "checks.headings" for n in (
            "_KW_TYPE", "_HEADING_TYPES", "_VI_NAME", "_KW_ALT", "_HNUM", "_HEAD_LINE", "_HEAD_ANY",
            "_HEAD_INLINE", "_hvalue", "heading_list", "_hsort", "display_heading", "_hstr",
            "compare_headings", "_has_heading")},
        **{n: "checks.loops" for n in (
            "LOOP_MIN_UNIT", "LOOP_MAX_UNIT", "LOOP_MIN_REPS", "LINE_MIN_CHARS", "LINE_MIN_COUNT",
            "PUNCT_LOOP_MIN_CHARS", "_LOOP_PATTERNS", "_zero_run_re", "_periodic_regions", "_loop_canon",
            "_loop_regions", "_LINE_LABEL", "_COLUMNAR", "_label_core", "_source_units", "find_loops")},
        **{n: "checks.lexical" for n in (
            "_VI_CHARS", "_WORD", "_vi_word_ratio", "_untranslated", "_NEG_EXCLUDE_EN", "_NEG_EN",
            "_NEG_EXCLUDE_VI", "_NEG_VI", "negation_count", "_MODAL", "modal_count")},
        **{n: "steps.s10_doc_check" for n in ("_length_info", "audit_plain_output", "_piece_key", "check_document")},
    }),
    "context": ("context.common", {
        **{n: "context.source_index" for n in (
            "_DEF_HEAD", "_DEF_INTRO", "_ITEM_LABEL", "_HEADER_LINE", "_DATE_LINE", "_is_header_line",
            "_QUOTED_TERM", "_DEF_VERB", "_TERM_TAIL", "_FUNC_END", "_FUNC_START", "_INLINE", "_INLINE_SPLIT",
            "_QUOTED_ANY", "_clean_term", "_valid_term", "_paren_depth_ok", "_article_item_term",
            "_sentence_start", "_inline_terms", "_REF", "_NUM", "_DOC_WORDS_EN", "_STOP_SCAN_EN",
            "_DOC_WORDS_VI", "_AMEND_AFTER", "_AMEND_BEFORE", "_qualifier", "_Heading", "_Definition",
            "SourceIndex")},
        "Memory": "context.memory", "Glossary": "context.glossary",
        **{n: "context.evidence" for n in (
            "_opt", "_unit_view", "_sentences", "_tail_fit", "_history_text", "_truncate_title", "_fit",
            "select_evidence_detailed", "select_evidence", "_HEADERS", "render_background", "split_terms")},
    }),
    "trace": ("tracing.writer", {
        **{n: "core.hashing" for n in ("utc_now", "sha256_text", "sha256_file", "_MODEL_FILE_EXT",
                                       "model_sha256", "code_sha256", "write_json", "read_jsonl")},
        "ResumeCache": "tracing.cache",
        "make_trace_record": "tracing.records", "environment_text": "tracing.records",
        **{n: "tracing.smoke_table" for n in ("MARKER_EOS_CODES", "TABLE_NUMBER_HARD", "TABLE_NUMBER_WARN",
                                            "_code", "_summ", "smoke_row", "write_smoke_table")},
    }),
    "runtime": ("engine.doc_job", {
        **{n: "inputs.sources" for n in ("DIRECTIONS", "DocItem", "norm_directions", "expand_rows",
                                     "prediction_relpath", "read_source")},
        "_code": "core.codes", "_dedupe": "core.codes", "_stats": "core.codes",
        "DOC_STATUSES": "core.codes", "SOURCE_KINDS": "core.codes",
        "MARKER_CODES": "steps.s08_recovery", "is_marker_failure": "steps.s08_recovery",
        "_prompt_template": "steps.s04_prompt", "_TERM_LINE": "steps.s04_prompt", "term_pair": "steps.s04_prompt",
        "_Candidate": "steps.s09_select",
        "_parallel": "engine.scheduler", "_Runner": "engine.scheduler",
        "_Run": "engine.run",
        **{n: "engine.api" for n in ("translate_document", "_load_glossary", "_manifest_info", "_fill_hashes",
                                     "README_STUB", "CHANGE_NOTES_STUB", "_write_stubs", "_write_model_manifest",
                                     "_check_run_dir", "run_manifest", "_write_roundtrip")},
    }),
    "backends.__init__": ("backends.base", {"make_backend": "backends.__init__", "BACKENDS": "backends.base"}),
    "backends.hf": ("backends.hf", {}),
    "backends.mock": ("backends.mock", {}),
    "sft": ("train.sft", {}),
    "train_sft": ("train.train_sft", {}),
}
SKIP_MODULES = {"__init__", "__main__", "cli", "backends.llamacpp"}   # rewritten by hand
VERBATIM = {"text_normalize": "core.text_normalize"}   # byte-identical copy of the team reference (test enforced)

# Path/name fixes applied after the split: (file, old, new). Each `old` must occur exactly once.
POST_EDITS = [
    ("steps/s01_parse_lines.py", "    from .config import RunConfig", "    from ..core.config import RunConfig"),
    ("engine/api.py", "        from .tok import HyTok", "        from ..core.tok import HyTok"),
    ("engine/api.py", "        from .backends import make_backend", "        from ..backends import make_backend"),
    ("engine/api.py", "`python -m lam_docmt run`", "`python -m docmt run`"),
    ("backends/__init__.py", '''    if name == "llamacpp":
        from .llamacpp import LlamaCppBackend
        return LlamaCppBackend(getattr(cfg, "server_url", "http://127.0.0.1:8099"), cfg, tok, **kw)
''', ""),
    ("backends/__init__.py", "from .base import BACKENDS\n",
     "from .base import BACKENDS, Backend\n\n__all__ = [\"BACKENDS\", \"Backend\", \"make_backend\"]\n"),
    ("backends/base.py", 'BACKENDS = ("llamacpp", "hf", "mock")', 'BACKENDS = ("hf", "mock")'),
    ("backends/base.py", "- llamacpp: llama.cpp server, POST /completion (GGUF served elsewhere), threads per request.\n", ""),
    ("core/config.py", '    backend: str = "llamacpp"            # "llamacpp" | "hf" | "mock"',
     '    backend: str = "hf"                  # "hf" | "mock"'),
    ("core/config.py", "    model: str = \"\"                      # HF dir (hf) or GGUF path served by llama-server (llamacpp)",
     "    model: str = \"\"                      # HF model dir (hf backend)"),
    ("core/config.py", '    server_url: str = "http://127.0.0.1:8099"',
     '    server_url: str = "http://127.0.0.1:8099"   # unused since llama.cpp was dropped; kept so old run_config.json files load'),
    ("core/hashing.py", '    """sha256 of every lam_docmt/**/*.py file plus a combined hash."""',
     '    """sha256 of every docmt/**/*.py file plus a combined hash."""'),
    ("core/hashing.py", "    root = Path(pkg_dir) if pkg_dir else Path(__file__).resolve().parent\n",
     "    root = Path(pkg_dir) if pkg_dir else Path(__file__).resolve().parents[1]\n"),
]
# runtime Components: lazy import table now points at the step modules
COMPONENTS = {
    "parse_document": "steps.s01_parse", "split_long_nodes": "steps.s03_split", "pack_units": "steps.s03_pack",
    "render_payload": "steps.s03_pack", "parse_output": "steps.s06_parse_output", "output_budget": "steps.s05_budget",
    "recovery_units": "steps.s08_recovery", "marker_namespace": "steps.s03_pack",
    "check_unit": "steps.s07_validate_unit", "check_document": "steps.s10_doc_check",
    "SourceIndex": "context.source_index", "Memory": "context.memory", "Glossary": "context.glossary",
    "select_evidence": "context.evidence", "render_background": "context.evidence",
}


def modname(path: Path, root: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    return ".".join(rel.parts)


def stmt_names(node) -> list:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        tg = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [n.id for t in tg for n in ast.walk(t) if isinstance(n, ast.Name)]
    if isinstance(node, ast.Try):  # parse.py: TABLE_CAP
        return [n.id for b in node.body + [h for x in node.handlers for h in x.body]
                for n in ast.walk(b) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)]
    return []


def _annotations(node) -> set:
    """ids of nodes that only appear inside annotations (not evaluated: `from __future__ import annotations`)."""
    skip = set()
    for n in ast.walk(node):
        anns = []
        if isinstance(n, ast.arg) and n.annotation is not None:
            anns.append(n.annotation)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.returns is not None:
            anns.append(n.returns)
        if isinstance(n, ast.AnnAssign):
            anns.append(n.annotation)
        for a in anns:
            skip.update(id(x) for x in ast.walk(a))
    return skip


def used_names(node) -> set:
    skip = _annotations(node)
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and id(n) not in skip} | \
           {n.value.id for n in ast.walk(node)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and id(n) not in skip}


def resolve_rel(mod: str, level: int, target: str | None) -> str:
    parts = mod.split(".")
    base = parts[: len(parts) - level] if level else []
    if mod.endswith("__init__"):
        base = parts[: len(parts) - level]  # package __init__: level 1 = the package itself
    return ".".join([p for p in base if p != "__init__"] + ([target] if target else []))


def main(src_root: Path, out_root: Path):
    sources = {}
    for p in sorted(src_root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        sources[modname(p, src_root)] = p.read_text(encoding="utf-8")

    defined = {}            # (module, name) -> target module
    imported = {}           # (module, name) -> (other module, original name)
    external = defaultdict(list)   # module -> [(import stmt source, provided names)]
    body = defaultdict(list)       # target -> [(source module, lineno, text, node)]
    header = {}
    for mod, text in sources.items():
        if mod in SKIP_MODULES:
            continue
        tree = ast.parse(text)
        if mod in VERBATIM:
            for node in tree.body:
                for n in stmt_names(node):
                    defined[(mod, n)] = VERBATIM[mod]
            continue
        default, over = PLAN[mod]
        lines = text.splitlines(keepends=True)
        for i, node in enumerate(tree.body):
            if i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                header[mod] = ast.get_source_segment(text, node)
                continue
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                continue
            if isinstance(node, ast.ImportFrom) and (node.level or (node.module or "").startswith("lam_docmt")):
                pkg_mod = resolve_rel(mod, node.level, node.module) if node.level else node.module[len("lam_docmt."):]
                for a in node.names:
                    if node.module is None:          # "from . import Backend" inside backends/*
                        imported[(mod, a.asname or a.name)] = (pkg_mod + ".__init__" if pkg_mod else "__init__", a.name)
                    else:
                        imported[(mod, a.asname or a.name)] = (pkg_mod, a.name)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                provided = [(a.asname or a.name).split(".")[0] for a in node.names]
                external[mod].append((ast.get_source_segment(text, node), provided))
                continue
            names = stmt_names(node)
            tgt = default
            for n in names:
                if n in over:
                    tgt = over[n]
            if tgt is None:
                continue
            start = node.lineno - 1
            if getattr(node, "decorator_list", None):
                start = min(d.lineno for d in node.decorator_list) - 1
            # keep comment lines directly above the statement
            while start > 0 and lines[start - 1].lstrip().startswith("#"):
                start -= 1
            seg = "".join(lines[start: node.end_lineno])
            body[tgt].append((mod, node.lineno, seg, node))
            for n in names:
                defined[(mod, n)] = tgt

    def resolve(mod, name):
        seen = set()
        while (mod, name) not in defined:
            if (mod, name) in seen or (mod, name) not in imported:
                return None
            seen.add((mod, name))
            mod, name = imported[(mod, name)]
            if mod in sources and mod not in SKIP_MODULES and (mod, name) in defined:
                break
        return defined.get((mod, name)), name

    problems, deps = [], defaultdict(set)
    files = {}
    for tgt, stmts in body.items():
        own = {n for (m, _, _, node) in stmts for n in stmt_names(node)}
        need, ext = {}, []
        mods = sorted({m for m, *_ in stmts})
        for m, _, _, node in stmts:
            for n in used_names(node):
                r = resolve(m, n)
                if r is None:
                    continue
                where, orig = r
                if where == tgt:
                    if n not in own:
                        problems.append(f"{tgt}: {n} resolves to itself but is not defined there")
                    continue
                if n in need and need[n] != (where, orig):
                    problems.append(f"{tgt}: name clash {n}: {need[n]} vs {(where, orig)}")
                need[n] = (where, orig)
                deps[tgt].add(where)
        for n in own:
            if n in need:
                problems.append(f"{tgt}: {n} both defined and imported")
        used_all = set().union(*(used_names(node) for *_, node in stmts))
        for m in mods:
            for src, provided in external.get(m, []):
                if any(p in used_all for p in provided) and src not in ext:
                    ext.append(src)
        by_mod = defaultdict(list)
        for n, (where, orig) in sorted(need.items()):
            by_mod[where].append(n if n == orig else f"{orig} as {n}")
        depth = tgt.count(".")
        imp = []
        for where, names in sorted(by_mod.items()):
            w = where[: -len(".__init__")] if where.endswith(".__init__") else where
            same_pkg = w.rsplit(".", 1)[0] == tgt.rsplit(".", 1)[0] and "." in w
            ref = "." + w.rsplit(".", 1)[1] if same_pkg else "." * (depth + 1) + w
            imp.append(f"from {ref} import {', '.join(sorted(names))}")
        origin = ", ".join(f"lam_docmt/{m.replace('.', '/')}.py" for m in mods)
        owned = [m for m in mods if PLAN[m][0] == tgt and m in header]
        if owned:
            text = header[owned[0]] + f"\n# Split from {origin} (statements verbatim; tools/split_from_lam_docmt.py).\n"
        else:
            text = f'"""Split from {origin} (statements verbatim; tools/split_from_lam_docmt.py)."""\n'
        text += "from __future__ import annotations\n\n"
        if ext:
            text += "\n".join(sorted(ext, key=lambda s: (s.startswith("from"), s))) + "\n\n"
        if imp:
            text += "\n".join(imp) + "\n\n"
        ordered = sorted(stmts, key=lambda s: (mods.index(s[0]), s[1]))
        text += "\n\n".join(seg.rstrip("\n") for _, _, seg, _ in ordered) + "\n"
        files[tgt] = (text, header)
    # import cycles between target modules (module-level imports would fail)
    def cyc(a, path):
        for b in deps.get(a, ()):
            if b == path[0]:
                problems.append("cycle: " + " -> ".join(path + [b]))
            elif b not in path and len(path) < 6:
                cyc(b, path + [b])
    for t in deps:
        cyc(t, [t])
    if problems:
        print("\n".join(sorted(set(problems))))
        sys.exit(1)
    for tgt, (text, _) in files.items():
        p = out_root / (tgt.replace(".", "/") + ".py")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        print("wrote", p.relative_to(out_root), len(text.splitlines()), "lines")
    for mod, tgt in VERBATIM.items():
        (out_root / (tgt.replace(".", "/") + ".py")).write_bytes((src_root / (mod + ".py")).read_bytes())
    for rel, old, new in POST_EDITS:
        p = out_root / rel
        t = p.read_text(encoding="utf-8")
        assert t.count(old) == 1, (rel, old)
        p.write_text(t.replace(old, new), encoding="utf-8")
    p = out_root / "engine/doc_job.py"
    t = p.read_text(encoding="utf-8")
    for name, where in COMPONENTS.items():
        old = None
        for line in t.splitlines():
            if line.strip().startswith(f'"{name}": ("lam_docmt.'):
                old = line
        assert old, name
        t = t.replace(old, f'    "{name}": ("docmt.{where}", "{name}"),')
    assert '"lam_docmt.' not in t
    p.write_text(t, encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))

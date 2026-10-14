import json
import os
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

# Test inputs (source-side dev data + Hy-MT2 tokenizer) live outside pipeline-v1:
# default <repo>/task 3-10/CHUNG/inputs, override with DOCMT_TEST_INPUTS=/path/to/inputs.
INPUTS = Path(os.environ.get("DOCMT_TEST_INPUTS") or CODE.parent / "task 3-10" / "CHUNG" / "inputs")
if not INPUTS.is_dir():
    pytest.exit(f"test inputs not found: {INPUTS} (set DOCMT_TEST_INPUTS)", returncode=4)
TOKENIZER = INPUTS / "hy_a1_tokenizer" / "tokenizer.json"
SMOKE6 = INPUTS / "smoke6_doc_direction_manifest.jsonl"
PILOT16 = INPUTS / "pilot16_doc_direction_manifest.jsonl"
DEV_MANIFEST = INPUTS / "release_v2" / "raw_dev_manifest.jsonl"   # paths relative to release_v2/
GLOSSARY = INPUTS / "release_v2" / "glossary_provenance.jsonl"


@pytest.fixture(scope="session")
def tok():
    from docmt.core.tok import HyTok
    return HyTok(TOKENIZER)


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.fixture(scope="session")
def smoke6():
    """[(row, source_text)] for the six historical doc-directions (source side only)."""
    rows = read_jsonl(SMOKE6)
    return [(r, (INPUTS / r["source_path"]).read_text(encoding="utf-8")) for r in rows]


@pytest.fixture(scope="session")
def dev84():
    """[(sample_id, direction, src_lang, tgt_lang, source_text)] for all 42 dev docs x 2 directions."""
    out = []
    for r in read_jsonl(DEV_MANIFEST):
        for src, tgt in (("en", "vi"), ("vi", "en")):
            text = (INPUTS / "release_v2" / r[f"{src}_path"]).read_text(encoding="utf-8")
            out.append((r["sample_id"], f"{src}-{tgt}", src, tgt, text))
    return out

"""docmt/core/text_normalize.py must stay a byte-identical copy of the team reference module."""
import hashlib
import importlib.util

from conftest import INPUTS

REF = INPUTS / "normalization_reference"


def _sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_copy_is_identical():
    from pathlib import Path
    mine = Path(__file__).resolve().parents[1] / "docmt" / "core" / "text_normalize.py"
    assert _sha(mine) == _sha(REF / "text_normalize.py")


def test_reference_cases_pass_on_copy():
    from docmt.core.text_normalize import normalize
    spec = importlib.util.spec_from_file_location("ref_cases", REF / "test_normalize.py")
    src = (REF / "test_normalize.py").read_text(encoding="utf-8")
    import unicodedata as ud
    ns = {"ud": ud}
    # Only take the CASES table from the reference test, then check it against our copy.
    start = src.index("CASES = {")
    end = src.index("\n}\n", start) + 3
    exec(src[start:end], ns)
    bad = {k: (normalize(k), v) for k, v in ns["CASES"].items() if normalize(k) != v}
    assert not bad, bad

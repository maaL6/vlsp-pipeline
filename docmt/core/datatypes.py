"""Shared data types. Every module of lam_docmt exchanges these objects (see SPEC.md)."""
# Split from lam_docmt/types.py (statements verbatim; tools/split_from_lam_docmt.py).
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Node kinds produced by parse.py. Unknown structure is never dropped: it becomes
# "paragraph" (or "unknown" with low confidence) and is still translated.
NODE_KINDS = (
    "separator",        # line without letters/digits worth translating (----, ****, ____): copied
    "header_cell",      # one cell of a multi-column header line (agency | national motto ...)
    "title",            # document / attachment title lines (usually centred, upper-case)
    "part", "chapter", "section",  # Part/Phần, Chapter/Chương, Section/Mục headings
    "article",          # Article N / Điều N heading line (may carry body text on the same line)
    "clause",           # numbered clause "1.", "1-", "1)", "2.1."
    "point",            # lettered point "a)", "đ)", "b."
    "list_item",        # "-", "+", "•", "*" bullets
    "paragraph",        # generic prose (soft-wrapped lines are unwrapped into one payload line)
    "table_cell",       # one cell of a column-aligned (plain-text) or pipe table row
    "signature",        # signature-block line other than the signer's name (titles, "(Signed)")
    "signature_name",   # signer's name line with high confidence: copied verbatim (protected)
    "attachment_title", # first heading line of an attachment / appendix / annex scope
    "note",             # parenthetical notes like "(Promulgated together with ...)"
    "unknown",          # generic node, low confidence, still translated
)

@dataclass
class Node:
    node_id: str                 # "n00042" (doc order); split pieces get "n00042.1", "n00042.2" ...
    kind: str                    # one of NODE_KINDS
    raw_start: int               # offset in the RAW source string (inclusive)
    raw_end: int                 # exclusive. Nodes partition [0, len(src)) exactly, in order
    lead: str                    # raw prefix kept verbatim in the output (indentation / column gap)
    content_raw: str             # raw content (lead + content_raw + trail == src[raw_start:raw_end])
    trail: str                   # raw suffix kept verbatim (trailing spaces, newline(s), blank lines)
    payload: str = ""            # model input = normalize(unwrapped content_raw); "" when not translatable
    translatable: bool = True    # False -> output keeps content_raw verbatim
    scope_id: str = "main"       # "main" | "attachment_1" | "appendix_1" | "quote_3" ...
    path: list = field(default_factory=list)       # e.g. ["Chapter II", "Article 6", "1."]
    confidence: float = 1.0      # parser confidence in `kind`
    protected: dict = field(default_factory=dict)  # see SPEC.md "Protected items"
    table: Optional[dict] = None # {"group": "t3", "row": 5, "col": 2, "numeric": False, "certain": False}
    flags: list = field(default_factory=list)      # e.g. ["broken_table", "page_break_join"]

    @property
    def raw(self) -> str:
        return self.lead + self.content_raw + self.trail

    def to_dict(self):
        return asdict(self)

@dataclass
class Unit:
    unit_id: str                 # "<doc_key>:u0007" ; recovery children "<parent>.r1", "<parent>.r2" ...
    doc_id: str                  # sample_id, e.g. "Dich_vu_phap_ly/136"
    scope_id: str
    kind: str                    # "text" | "table" | "signature" | "recovery" | "single"
    node_ids: list               # translatable node ids, in order
    markers: list                # marker strings aligned with node_ids ([] for single-node units)
    payload: str                 # the {source} text sent to the model
    source_tokens: int           # tokens of payload, markers included
    raw_start: int
    raw_end: int
    raw_sha256: str              # sha256 of src[raw_start:raw_end] (utf-8)
    split_from: Optional[str] = None

    def to_dict(self):
        return asdict(self)

@dataclass
class EvidenceItem:
    kind: str                    # "title_path" | "definition" | "reference" | "glossary" | "history"
    origin: str                  # "source_title" | "source_path" | "source_definition" | "source_reference"
                                 # | "train_glossary" | "accepted_history"
    text: str
    tokens: int
    ref: dict = field(default_factory=dict)  # node ids / glossary entry / history unit id

    def to_dict(self):
        return asdict(self)

@dataclass
class GenRequest:
    request_id: str
    prompt: str                  # fully rendered chat prompt (BOS ... <|hy_Assistant|>)
    max_new_tokens: int
    repetition_penalty: float = 1.05
    meta: dict = field(default_factory=dict)

@dataclass
class GenResult:
    request_id: str
    text: str                    # decoded output, special tokens removed, EOS excluded
    eos: bool                    # True only if the model emitted EOS within max_new_tokens
    stop_type: str               # "eos" | "limit" | "error"
    input_tokens: int
    output_tokens: int
    seconds: float = 0.0
    error: Optional[str] = None  # operational error text (network/OOM after retries)
    backend: str = ""

@dataclass
class ParseResult:
    segments: dict               # node_id -> translated text (stripped)
    hard: list                   # marker_missing:<id>, marker_duplicate:<id>, marker_order, marker_unexpected:<m>,
                                 # unassigned_text, segment_empty:<id>, empty_output
    extra: dict = field(default_factory=dict)

@dataclass
class UnitCheck:
    hard: list                   # hard failure codes (deterministic, see SPEC.md)
    warnings: list               # warning codes (heuristic, need review)
    details: dict = field(default_factory=dict)

@dataclass
class Attempt:
    unit: Unit
    request: GenRequest
    result: GenResult
    parsed: ParseResult
    check: UnitCheck
    evidence: list = field(default_factory=list)
    attempt: int = 0             # 0 first pass, 1 quality-recovery round
    retry_reason: Optional[str] = None

@dataclass
class DocumentResult:
    output_path: str
    status: str                  # "completed" | "completed_with_unresolved" | "failed"
    diagnostics: dict = field(default_factory=dict)

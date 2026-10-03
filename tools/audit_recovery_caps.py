"""Source-only check of both recovery modes on every planned smoke6 unit."""
import hashlib
import json
import sys
from pathlib import Path

V1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V1))
from docmt.core.config import RunConfig
from docmt.core.tok import HyTok
from docmt.steps.s01_parse import parse_document
from docmt.steps.s03_pack import marker_namespace, pack_units
from docmt.steps.s03_split import split_long_nodes
from docmt.steps.s08_recovery import recovery_units

inputs = V1.parent / "task 3-10/CHUNG/inputs"
tok = HyTok(inputs / "hy_a1_tokenizer/tokenizer.json")
cfg = RunConfig()
parents = children = 0
errors = []
for line in (inputs / "smoke6_doc_direction_manifest.jsonl").read_text().splitlines():
    row = json.loads(line)
    source = (inputs / row["source_path"]).read_bytes().decode()
    nodes = split_long_nodes(parse_document(source, row["sample_id"], row["src_lang"], tok), tok, cfg)
    ns = marker_namespace(source)
    for unit in pack_units(nodes, row["sample_id"], tok, cfg, ns):
        parents += 1
        for independent in (True, False):
            by_id = {node.node_id:node for node in nodes}
            for child in recovery_units(unit, by_id, tok, cfg, ns, independent):
                children += 1
                start,end = child.raw_start,child.raw_end
                if (child.source_tokens > cfg.recovery_cap or
                    hashlib.sha256(source[start:end].encode()).hexdigest() != child.raw_sha256):
                    errors.append(dict(doc_id=row["sample_id"],unit_id=child.unit_id,
                                       source_tokens=child.source_tokens,independent=independent))
payload = (V1 / "tests/fixtures/recovery_369_source.txt").read_text()
from docmt.core.datatypes import Node
node = Node("fixture", "paragraph", 0, len(payload), "", payload, "", payload)
unit, = pack_units([node], "fixture", tok, cfg, "SEG")
by_id = {node.node_id:node}
fixed = recovery_units(unit, by_id, tok, cfg, "SEG", True)
report = dict(status="completed" if not errors else "failed", all_checks_passed=not errors,
              source_only=True, inference_performed=False, parent_units=parents, modes_checked=2,
              recovery_children=children, errors=errors, regression_original_tokens=tok.count(payload),
              regression_fixed_piece_tokens=[c.source_tokens for c in fixed],
              source_span_preserved="".join(by_id[c.node_ids[0]].raw for c in fixed)==payload)
out = V1 / "reports/recovery_cap_fix_audit.json"
out.write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
raise SystemExit(bool(errors) or not report["source_span_preserved"])

"""Independently re-read authoritative train rows and verify each derivative's target/context/labels."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path

V1 = Path(__file__).resolve().parents[1]            # pipeline-v1 (outputs are written here)
REPORT = V1.parent / "report_lam_3_10"              # read-only: token_audit.py of the delivered package


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    inputs = V1.parent / "task 3-10/CHUNG/inputs"
    root = V1 / "train" / "data"
    spec = importlib.util.spec_from_file_location("audit_native", REPORT / "01_prompt_contract_for_Linh/token_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = module.Ctx(inputs / "hy_a1_tokenizer", inputs)
    original = [json.loads(line) for line in (inputs / "release_v2/sft_train.jsonl").open()]
    splits = json.loads((inputs / "release_v2/doc_splits.json").read_text())
    errors = []
    pairs = collections.defaultdict(dict)
    counts = collections.Counter()
    checked = 0
    for line in (root / "hy_context_pilot.jsonl").open():
        row = json.loads(line)
        checked += 1
        raw = original[row["release_line"] - 1]
        try:
            assert splits[raw["meta"]["sample_id"]] == "train"
            assert row["sample_id"] == raw["meta"]["sample_id"]
            assert row["target"] == raw["output"]
            assert digest(raw["input"]) == row["source_sha256"]
            assert digest(raw["output"]) == row["target_sha256"]
            assert row["direction"] == raw["src_lang"] + "-" + raw["tgt_lang"]
            assert row["direction"] not in pairs[row["payload_id"]]
            pairs[row["payload_id"]][row["direction"]] = raw
            user = row["user"]
            source = module.normalize(raw["input"])
            assert user.endswith(source)
            if row["mode"] == "source_context":
                mapping = row["mapping"]
                parent = original[mapping["parent_line"] - 1]
                assert parent["meta"]["sample_id"] == row["sample_id"]
                assert parent["src_lang"] == raw["src_lang"]
                assert digest(parent["input"]) == mapping["parent_source_sha256"]
                psrc = module.normalize(parent["input"])
                assert digest(psrc) == mapping["normalized_parent_sha256"]
                assert psrc.count(source) == 1
                assert psrc[mapping["source_start"]:mapping["source_end"]] == source
                background = user.split("[Background Information]\n", 1)[1].split("\n\nPlease translate", 1)[0]
                assert ctx.hy.count(background) <= 512
                for start, end in mapping["context_spans"]:
                    assert 0 <= start < end <= mapping["source_start"]
                    # Payload normalisation only unwraps whitespace; compare actual source words.
                    excerpt = " ".join(psrc[start:end].split())
                    assert excerpt in " ".join(background.split())
            else:
                assert user == module.prompts.build_user(source, raw["tgt_lang"])
            prefix = ctx.hy.encode(ctx.chat.infer(user))
            full = ctx.hy.encode(ctx.chat.train(user, raw["output"]))
            target = ctx.hy.encode(raw["output"] + module.prompts.EOS)
            assert full == prefix + target + ctx.hy.encode(module.prompts.END_NO_GEN)
            assert len(full) <= 4096 and len(full) == row["sequence_tokens"]
            assert ctx.hy.decode(target, skip_special_tokens=False) == raw["output"] + module.prompts.EOS
            counts[row["direction"]] += 1
        except (AssertionError, KeyError, IndexError, ValueError) as exc:
            errors.append(dict(payload_id=row["payload_id"], direction=row["direction"], error=type(exc).__name__))
    for payload_id, pair in pairs.items():
        if set(pair) != {"en-vi", "vi-en"} or pair["en-vi"]["input"] != pair["vi-en"]["output"] or \
                pair["en-vi"]["output"] != pair["vi-en"]["input"]:
            errors.append(dict(payload_id=payload_id, error="common_aligned_pair_mismatch"))
    report = dict(status="completed" if not errors else "failed", rows=checked, pairs=len(pairs),
                  all_checks_passed=not errors, errors=errors, rows_by_direction=dict(counts),
                  payload_file_sha256=hashlib.sha256((root / "hy_context_pilot.jsonl").read_bytes()).hexdigest(),
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  review_kind="independent_programmatic_recheck_against_release_rows", human_review="not_checked")
    (root / "independent_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

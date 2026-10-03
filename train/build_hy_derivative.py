"""Train-only paired HY pilot. Context requires a unique exact source span in a containing view.

Never joins aligned targets or infers adjacency from section/paragraph indices. Target text is
copied unchanged. Ambiguous/missing mapping falls back to direct replay. Normalized offsets
refer to the explicitly hashed containing SFT source view, not to an unverified raw CSV span.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

V1 = Path(__file__).resolve().parents[1]            # pipeline-v1 (outputs are written here)
REPORT = V1.parent / "report_lam_3_10"              # read-only: token_audit.py of the delivered package
sys.path.insert(0, str(V1))
from docmt.core.text_normalize import normalize
from docmt.steps.s01_parse import parse_document
from docmt.steps.s04_prompt import EOS, build_user


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    inputs = V1.parent / "task 3-10/CHUNG/inputs"
    out = V1 / "train" / "data"
    out.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("hy_audit", REPORT / "01_prompt_contract_for_Linh/token_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ctx = module.Ctx(inputs / "hy_a1_tokenizer", inputs)
    split = json.loads((inputs / "release_v2/doc_splits.json").read_text())
    train = inputs / "release_v2/sft_train.jsonl"
    rows = []
    paired = collections.defaultdict(dict)
    parents = collections.defaultdict(list)
    duplicate = 0
    for line, text in enumerate(train.open(), 1):
        row = json.loads(text)
        assert split[row["meta"]["sample_id"]] == "train"
        row["line"] = line
        row["normalized_source"] = normalize(row["input"])
        rows.append(row)
        meta = row["meta"]
        direction = row["src_lang"] + "-" + row["tgt_lang"]
        en, vi = (row["input"], row["output"]) if direction == "en-vi" else (row["output"], row["input"])
        key = (meta["sample_id"], meta["view"], meta.get("section_id", ""), sha(en), sha(vi))
        if direction in paired[key]:
            duplicate += 1
        paired[key].setdefault(direction, row)
        if meta["view"] in ("article", "document", "full_document"):
            parents[(meta["sample_id"], direction)].append(row)
    groups = collections.defaultdict(list)
    for key, value in paired.items():
        if set(value) == {"en-vi", "vi-en"}:
            groups[(value["en-vi"]["meta"]["category"], key[0])].append((key, value))
    rng = random.Random(42)
    for value in groups.values():
        rng.shuffle(value)
    group_keys = sorted(groups)
    rng.shuffle(group_keys)
    parsed = {}
    def evidence(row):
        source = row["normalized_source"]
        candidates = [p for p in parents[(row["meta"]["sample_id"], row["src_lang"]+"-"+row["tgt_lang"])]
                      if p["normalized_source"] != source and p["normalized_source"].count(source) == 1]
        for parent in sorted(candidates, key=lambda p: (len(p["normalized_source"]), p["line"])):
            text = parent["normalized_source"]
            start = text.index(source)
            end = start + len(source)
            if parent["line"] not in parsed:
                parsed[parent["line"]] = parse_document(text, parent["meta"]["sample_id"], parent["src_lang"], ctx.hy)
            nodes = parsed[parent["line"]]
            owners = {n.scope_id for n in nodes if n.raw_end > start and n.raw_start < end}
            if len(owners) != 1:
                continue
            scope = next(iter(owners))
            # Select complete preceding source nodes from the same legal scope, no target history.
            available = [n for n in nodes if n.scope_id == scope and n.raw_end <= start and n.payload.strip()]
            selected = []
            for node in reversed(available):
                trial = [node] + selected
                background = "Source context from the same legal scope:\n" + "\n".join(n.payload for n in trial)
                if ctx.hy.count(background) <= 512:
                    selected = trial
                if len(selected) >= 3:
                    break
            if not selected:
                continue
            background = "Source context from the same legal scope:\n" + "\n".join(n.payload for n in selected)
            mapping = dict(status="verified_unique_exact_normalized_source_span", scope_id=scope,
                           parent_line=parent["line"], parent_source_sha256=sha(parent["input"]),
                           normalized_parent_sha256=sha(text), source_start=start, source_end=end,
                           coordinate_space="normalized containing SFT source view", context_origin="train_source_only",
                           context_spans=[[n.raw_start, n.raw_end] for n in selected],
                           evidence_tokens=ctx.hy.count(background))
            assert text[start:end] == source and all(e <= start for s, e in mapping["context_spans"])
            return background, mapping
        return "", dict(status="no_verified_context_mapping", context_origin="none")
    manifest = []
    derivative = []
    dropped = []
    examples = []
    accepted_pairs = 0
    contexts = 0
    while group_keys and accepted_pairs < 6000:
        remaining = []
        for group in group_keys:
            if accepted_pairs >= 6000:
                break
            if not groups[group]:
                continue
            key, pair = groups[group].pop()
            want_context = contexts < 4200
            payload_id = sha(json.dumps(key, ensure_ascii=False))[:24]
            built = []
            for direction in ("en-vi", "vi-en"):
                row = pair[direction]
                background, mapping = evidence(row) if want_context else ("", dict(status="direct_replay", context_origin="none"))
                user = build_user(row["normalized_source"], row["tgt_lang"], background=background)
                built.append([row, user, background, mapping])
            # Keep the same direct/context mode for both directions of a common payload.
            context_pair = want_context and all(x[2] for x in built)
            if not context_pair:
                for item in built:
                    item[1] = build_user(item[0]["normalized_source"], item[0]["tgt_lang"])
                    item[2] = ""
                    item[3] = dict(status="direct_replay_fallback", context_origin="none")
            encoded = []
            valid = True
            for row, user, background, mapping in built:
                full = ctx.chat.train(user, row["output"])
                ids = ctx.hy.encode(full)
                prefix = ctx.hy.encode(ctx.chat.infer(user))
                target = ctx.hy.encode(row["output"] + EOS)
                tail = ctx.hy.encode(module.prompts.END_NO_GEN)
                assert ids == prefix + target + tail
                assert ctx.hy.decode(target, skip_special_tokens=False) == row["output"] + EOS
                if len(ids) > 4096:
                    valid = False
                    dropped.append(dict(payload_id=payload_id, direction=row["src_lang"]+"-"+row["tgt_lang"],
                                        reason="sequence_exceeds_4096_target_not_truncated", tokens=len(ids), line=row["line"]))
                encoded.append(dict(sequence_tokens=len(ids), prompt_tokens=len(prefix), label_tokens=len(target)))
            if valid:
                accepted_pairs += 1
                contexts += int(context_pair)
                for (row, user, background, mapping), counts in zip(built, encoded):
                    direction = row["src_lang"] + "-" + row["tgt_lang"]
                    meta = row["meta"]
                    record = dict(payload_id=payload_id, sample_id=meta["sample_id"], category=meta["category"],
                                  direction=direction, src_lang=row["src_lang"], tgt_lang=row["tgt_lang"],
                                  view=meta["view"], section_id=meta.get("section_id"), split="train", release_line=row["line"],
                                  mode="source_context" if context_pair else "direct", source_sha256=sha(row["input"]),
                                  target_sha256=sha(row["output"]), mapping=mapping, **counts)
                    manifest.append(record)
                    derivative.append(dict(**record, user=user, target=row["output"]))
                    if len(examples) < 4 or counts["sequence_tokens"] > examples[-1]["sequence_tokens"]:
                        examples.append(dict(payload_id=payload_id, direction=direction, user=user, target=row["output"], **counts))
                        examples = sorted(examples, key=lambda x:x["sequence_tokens"])
                        if len(examples) > 8:
                            examples = examples[:4] + examples[-4:]
            if groups[group]:
                remaining.append(group)
        group_keys = remaining
    for name, records in [("common_payload_manifest.jsonl", manifest), ("hy_context_pilot.jsonl", derivative),
                          ("dropped_rows.jsonl", dropped), ("train_smoke.jsonl", examples)]:
        (out / name).write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records), encoding="utf-8")
    directions = dict(collections.Counter(r["direction"] for r in manifest))
    modes = dict(collections.Counter(r["mode"] for r in manifest))
    report = dict(status="completed", owner="Lam", provenance="new local builder; not a recovered Claude or Khanh Linh artifact",
                  source_file=str(train), source_file_sha256=hashlib.sha256(train.read_bytes()).hexdigest(),
                  tokenizer_sha256=ctx.hy.sha256, template_sha256=sha(ctx.chat.source),
                  pairs=accepted_pairs, rows=len(manifest), rows_by_direction=directions,
                  actual_modes=modes, planned_context_fraction=0.7, actual_context_fraction=contexts/max(accepted_pairs,1),
                  documents=len({r["sample_id"] for r in manifest}),
                  category_counts=dict(collections.Counter(r["category"] for r in manifest)),
                  dropped_rows=len(dropped), deduplicated_same_direction_rows=duplicate,
                  no_eval_documents=all(split[r["sample_id"]] == "train" for r in manifest),
                  targets_unchanged=True, label_decode_all_passed=True, prefix_mask_all_passed=True,
                  max_sequence=4096, max_observed_sequence=max((r["sequence_tokens"] for r in manifest), default=0),
                  context_max_tokens=max((r["mapping"].get("evidence_tokens",0) for r in manifest),default=0),
                  target_history=False, train_status="not_started", independent_mapping_review="not_checked",
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.jsonl")})
    (out / "derivative_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ['files','category_counts']},indent=2),flush=True)


if __name__ == "__main__":
    main()

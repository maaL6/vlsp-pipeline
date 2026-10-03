"""Native HY assistant-only labels and a disposable LoRA forward/backward prerequisite check.

The smoke check does not produce a trained checkpoint. Full SFT requires an independently
audited train-only payload derivative, rather than inference smoke sources or dev targets.
"""
# Split from lam_docmt/sft.py (statements verbatim; tools/split_from_lam_docmt.py).
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from ..steps.s04_prompt import END_NO_GEN, EOS, render_chat

def encode_example(tokenizer, user, target, max_sequence=4096):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user}, {"role": "assistant", "content": target}],
        tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(text, add_special_tokens=False)
    prefix = tokenizer.encode(render_chat(user), add_special_tokens=False)
    target_ids = tokenizer.encode(target + EOS, add_special_tokens=False)
    tail = tokenizer.encode(END_NO_GEN, add_special_tokens=False)
    if ids != prefix + target_ids + tail:
        raise ValueError("native template prefix/target/EOS boundary mismatch")
    if tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != target + EOS:
        raise ValueError("decoded labels do not equal the original assistant target plus EOS")
    if len(ids) > max_sequence:
        raise ValueError(f"target would require truncation: {len(ids)} > {max_sequence}")
    labels = [-100] * len(prefix) + target_ids + [-100] * len(tail)
    return dict(input_ids=ids, labels=labels, prompt_tokens=len(prefix),
                target_tokens=len(target_ids), sequence_tokens=len(ids))

def smoke(model_path, samples_path, out_path):
    import torch
    import transformers
    import peft
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(42)
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", kind="disposable_forward_backward_smoke",
                  train_status="not_started", model=model_path,
                  samples_sha256=hashlib.sha256(Path(samples_path).read_bytes()).hexdigest(),
                  target_truncation=False, learning_rate=5e-5,
                  versions=dict(torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__),
                  samples=[])
    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                     default=lambda x: sorted(x) if isinstance(x, set) else str(x)), encoding="utf-8")
    save()
    try:
        tk = AutoTokenizer.from_pretrained(model_path)
        dtype = {"dtype": torch.float16} if int(transformers.__version__.split('.')[0]) >= 5 else {"torch_dtype": torch.float16}
        model = AutoModelForCausalLM.from_pretrained(model_path, **dtype).to("cuda")
        report["base_parameter_count"] = sum(p.numel() for p in model.parameters())
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        required = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        found = sorted({name.rsplit('.', 1)[-1] for name, module in model.named_modules()
                        if isinstance(module, torch.nn.Linear) and name.rsplit('.', 1)[-1] in required})
        if sorted(required) != found:
            raise ValueError(f"attention/MLP module audit failed: {found}")
        config = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
                            target_modules=required, task_type="CAUSAL_LM", bias="none")
        model = get_peft_model(model, config)
        model.train()
        report["lora"] = config.to_dict()
        report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows = [json.loads(x) for x in Path(samples_path).read_text().splitlines() if x.strip()]
        torch.cuda.reset_peak_memory_stats()
        for row in rows:
            encoded = encode_example(tk, row["user"], row["target"])
            model.zero_grad(set_to_none=True)
            ids = torch.tensor([encoded["input_ids"]], device="cuda")
            labels = torch.tensor([encoded["labels"]], device="cuda")
            start = time.time()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite loss")
            loss.backward()
            gradients = [p.grad for name, p in model.named_parameters() if p.requires_grad and p.grad is not None]
            finite = bool(gradients) and all(torch.isfinite(g).all().item() for g in gradients)
            nonzero = any(torch.count_nonzero(g).item() > 0 for g in gradients)
            if not (finite and nonzero):
                raise ValueError("LoRA gradients missing, nonfinite or all zero")
            report["samples"].append(dict(payload_id=row["payload_id"], direction=row["direction"],
                                          sequence_tokens=encoded["sequence_tokens"], loss=float(loss.detach()),
                                          gradient_tensors=len(gradients), finite_gradients=finite,
                                          nonzero_gradients=nonzero, seconds=round(time.time()-start, 3)))
            del ids, labels, loss
            save()
        report.update(status="completed", peak_vram_bytes=torch.cuda.max_memory_allocated(),
                      gpu=torch.cuda.get_device_name(0), all_checks_passed=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", all_checks_passed=False)
        raise
    finally:
        save()
    return report

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    smoke(args.model, args.samples, args.out)

if __name__ == "__main__":
    main()

"""Fresh HY context LoRA pilot. Resume is restricted to this run's own optimizer checkpoint."""
# Split from lam_docmt/train_sft.py (statements verbatim; tools/split_from_lam_docmt.py).
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import time
from pathlib import Path

from .sft import encode_example

def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--payloads", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", default="none", help="none or auto; auto only accepts this run's checkpoints")
    parser.add_argument("--max-updates", type=int, help="explicit diagnostic short run; never labelled full pilot")
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import Dataset

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    assert 32 % world == 0, "effective batch32 must divide world size"
    derivative_audit = json.loads(Path(args.audit).read_text())
    payload_sha = file_sha(args.payloads)
    assert derivative_audit["all_checks_passed"] and derivative_audit["payload_file_sha256"] == payload_sha
    model_sha = file_sha(Path(args.model) / "model.safetensors")
    assert model_sha == "5dc305ae6c33c15a8440a4a8f9183e411004079eb737bc2da417b6a932dead81"
    rows = [json.loads(line) for line in Path(args.payloads).read_text().splitlines() if line.strip()]
    assert rows and all(row["split"] == "train" for row in rows)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(run_id=args.run_id, initial_model_sha256=model_sha, payload_sha256=payload_sha,
                  rows=len(rows), max_sequence=4096, target_truncation=False, seed=42, epochs=1,
                  lora_r=32, lora_alpha=64, lora_dropout=0.05, learning_rate=5e-5,
                  warmup_ratio=0.03, effective_batch=32, microbatch=1, world_size=world,
                  gradient_accumulation_steps=32//world, precision="fp16", max_updates=args.max_updates,
                  context_history="source_only", fresh_lora=True,
                  code_sha256=file_sha(__file__), independent_audit_sha256=file_sha(args.audit),
                  planned_updates=args.max_updates or math.ceil(len(rows)/32),
                  assistant_only_loss="target plus native EOS; prompt/context/ASSISTANT/END8 masked",
                  kept_rows=len(rows), dropped_rows_report="derivative_audit.json and dropped_rows.jsonl",
                  checkpoint_schedule="middle and end of epoch", evaluation_schedule="separate fixed-dev inference",
                  timezone="UTC; team UTC+7")
    config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["config_sha256"] = config_sha
    config_path = output / "training_config.json"
    checkpoint = None
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous.get("config_sha256") != config_sha:
            raise ValueError("refusing cross-run/config/dataset optimizer resume")
        if args.resume == "auto":
            checkpoints = sorted(output.glob("checkpoint-*"), key=lambda p: int(p.name.split('-')[-1]))
            checkpoint = str(checkpoints[-1]) if checkpoints else None
        elif list(output.glob("checkpoint-*")):
            raise ValueError("existing optimizer checkpoint requires explicit --resume auto for this run")
    if rank == 0:
        config_path.write_text(json.dumps(config, indent=2))
    torch.manual_seed(42)
    tk = AutoTokenizer.from_pretrained(args.model)
    dtype_args = {"dtype": torch.float16} if int(transformers.__version__.split('.')[0]) >= 5 else {"torch_dtype": torch.float16}
    model = AutoModelForCausalLM.from_pretrained(args.model, **dtype_args)
    model.config.use_cache = False
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    found = {name.rsplit('.',1)[-1] for name, module in model.named_modules() if isinstance(module,torch.nn.Linear)}
    assert set(target_modules) <= found
    base_count = sum(p.numel() for p in model.parameters())
    model = get_peft_model(model, LoraConfig(r=32,lora_alpha=64,lora_dropout=0.05,
                                           target_modules=target_modules,task_type="CAUSAL_LM",bias="none"))
    class PayloadDataset(Dataset):
        def __len__(self):
            return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            encoded = encode_example(tk, row["user"], row["target"])
            return dict(input_ids=encoded["input_ids"], labels=encoded["labels"],
                        attention_mask=[1]*len(encoded["input_ids"]))
    def collate(batch):
        width = max(len(r["input_ids"]) for r in batch)
        return {key:torch.tensor([r[key]+[pad]*(width-len(r[key])) for r in batch],dtype=torch.long)
                for key,pad in [("input_ids",tk.pad_token_id),("labels",-100),("attention_mask",0)]}
    total = math.ceil(len(rows)/32)
    save_steps = max(1, math.ceil(total/2))
    class Status(TrainerCallback):
        def __init__(self):
            self.started = time.time()
            self.initial_step = 0
            self.logs = []
        def record(self, state, **extra):
            if not state.is_world_process_zero:
                return
            seconds = time.time()-self.started
            updates = state.global_step
            measured_updates = updates - self.initial_step
            report = dict(status="running", run_id=args.run_id, step=updates, planned_updates=state.max_steps,
                          utc_updated=datetime.datetime.now(datetime.timezone.utc).isoformat(), timezone="UTC; team UTC+7",
                          updates_per_second=measured_updates/max(seconds,1),
                          eta_seconds=(state.max_steps-updates)*seconds/measured_updates if measured_updates else None,
                          config_sha256=config_sha, payload_sha256=payload_sha,
                          base_parameter_count=base_count, total_parameters=sum(p.numel() for p in model.parameters()),
                          trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                          checkpoint_origin="fresh LoRA on merged A1; own-run optimizer resume only",
                          history=self.logs[-20:], **extra)
            (output/"train_status.json").write_text(json.dumps(report,indent=2))
        def on_train_begin(self, args, state, control, **kwargs):
            self.started = time.time()
            self.initial_step = state.global_step
            self.record(state)
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for key in ("loss", "grad_norm"):
                    if key in logs and not math.isfinite(float(logs[key])):
                        raise ValueError(f"nonfinite training {key}")
                self.logs.append(dict(step=state.global_step,**logs))
            self.record(state)
        def on_save(self, args, state, control, **kwargs):
            self.record(state,last_checkpoint=f"checkpoint-{state.global_step}")
            if state.is_world_process_zero:
                checkpoint_dir = output/f"checkpoint-{state.global_step}"
                manifest = dict(run_id=config["run_id"],step=state.global_step,config_sha256=config_sha,
                                files={str(p.relative_to(checkpoint_dir)):file_sha(p)
                                       for p in checkpoint_dir.rglob('*') if p.is_file()})
                (output/"checkpoint_manifest.json").write_text(json.dumps(manifest,indent=2))
    callback = Status()
    kwargs = dict(output_dir=str(output),num_train_epochs=1,learning_rate=5e-5,warmup_ratio=0.03,
                  per_device_train_batch_size=1,gradient_accumulation_steps=32//world,
                  fp16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={"use_reentrant":False},
                  logging_steps=5,save_strategy="steps",save_steps=save_steps,save_total_limit=3,
                  seed=42,data_seed=42,report_to=[],remove_unused_columns=False,ddp_find_unused_parameters=False,
                  optim="adamw_torch",lr_scheduler_type="linear",weight_decay=0.0,max_grad_norm=1.0,
                  dataloader_num_workers=0)
    if args.max_updates:
        kwargs["max_steps"] = args.max_updates
        kwargs["save_steps"] = min(save_steps,args.max_updates)
    trainer = Trainer(model=model,args=TrainingArguments(**kwargs),train_dataset=PayloadDataset(),
                      data_collator=collate,processing_class=tk,callbacks=[callback])
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(output/"adapter_final"))
        if rank == 0:
            report = json.loads((output/"train_status.json").read_text())
            report.update(status="completed", completion_scope="diagnostic_short_run" if args.max_updates else "full_pilot_epoch",
                          metrics=result.metrics,peak_vram_bytes=torch.cuda.max_memory_allocated())
            (output/"train_status.json").write_text(json.dumps(report,indent=2))
    except Exception as exc:
        if rank == 0:
            path = output/"train_status.json"
            report = json.loads(path.read_text()) if path.exists() else dict(run_id=args.run_id)
            report.update(status="failed",error=f"{type(exc).__name__}: {exc}")
            path.write_text(json.dumps(report,indent=2))
        raise

if __name__ == "__main__":
    main()

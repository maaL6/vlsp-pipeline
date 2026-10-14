"""HF transformers backend.

AutoModelForCausalLM (torch/transformers imported lazily in load()), fp16 on cuda by default, left
padding, greedy, repetition_penalty; one padded batch generate with max(max_new_tokens). The
per-request budget is enforced post-hoc: EOS index < own max_new_tokens -> eos True, else the
output is cut to its own budget and eos False. CUDA OOM -> halve the batch (persistently) and
count `oom`; OOM at batch size 1 -> operational error result. parameter_count is counted after
load (sum of numel over model.parameters(), shared/tied tensors counted once).
"""
from __future__ import annotations

import time

from .base import Backend
from ..core.datatypes import GenResult
from .batch_policy import next_chunk,padded_tokens

def postprocess_ids(seq, eos_ids, own_budget: int, pad_id=None):
    """(ids, eos) for one row of a batch generated with max(max_new_tokens) of the batch."""
    seq = list(seq)
    stop = next((i for i, t in enumerate(seq) if t in eos_ids), None)
    if stop is not None and stop < own_budget:
        return seq[:stop], True
    ids = seq[:own_budget]
    if pad_id is not None:
        while ids and ids[-1] == pad_id:
            ids.pop()
    return ids, False

def is_oom(exc) -> bool:
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()

class HFBackend(Backend):
    name = "hf"
    batching = True

    def __init__(self, model_path="", cfg=None, tok=None, *, device=None, dtype=None,
                 batch_size=None, revision=None):
        super().__init__()
        self.model_path = model_path or getattr(cfg, "model", "")
        self.cfg = cfg
        self.hytok = tok
        self.device = device or getattr(cfg, "hf_device", "cuda")
        self.dtype_name = dtype or getattr(cfg, "hf_dtype", "float16")
        self.max_batch = int(batch_size or getattr(cfg, "batch_size", 4) or 4)
        self.max_batch = min(self.max_batch,int((getattr(cfg,'extra',{}) or {}).get('batch_maximum',self.max_batch)))
        self.revision = revision
        self.model = None
        self.tokenizer = None
        self.parameter_count = None
        self.eos_ids: set = set()
        self.pad_id = None
        self.versions: dict = {}
        self.gpu_name = None

    # -- loading -----------------------------------------------------------------------------
    def prepare(self):
        self.load()

    def load(self):
        if self.model is not None:
            return
        import torch  # lazy: only this backend needs torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, self.dtype_name)
        major = int(transformers.__version__.split(".")[0])
        kw = {"dtype": dtype} if major >= 5 else {"torch_dtype": dtype}
        if self.revision:
            kw["revision"] = self.revision
        tk = AutoTokenizer.from_pretrained(self.model_path, **({"revision": self.revision} if self.revision else {}))
        tk.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(self.model_path, **kw).to(self.device)
        model.eval()
        self.parameter_count = int(sum(p.numel() for p in model.parameters()))
        eos = model.generation_config.eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        self.eos_ids = {int(e) for e in eos if e is not None}
        if tk.eos_token_id is not None:
            self.eos_ids.add(int(tk.eos_token_id))
        self.pad_id = tk.pad_token_id if tk.pad_token_id is not None else min(self.eos_ids)
        self.tokenizer, self.model = tk, model
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            self.gpu_name = torch.cuda.get_device_name(0)
            torch.cuda.reset_peak_memory_stats()

    # -- generation --------------------------------------------------------------------------
    def decode_config(self) -> dict:
        return {"backend": self.name, "do_sample": False, "num_beams": 1,
                "repetition_penalty": float(getattr(self.cfg, "repetition_penalty", 1.05)),
                "dtype": self.dtype_name, "padding": "left"}

    def generate(self, requests):
        reqs = list(requests)
        if not reqs:
            return []
        self.load()
        out = {}
        i = 0
        while i < len(reqs):
            chunk = next_chunk(reqs[i:],self.max_batch,
                               lambda p:len(self.tokenizer.encode(p,add_special_tokens=False)),
                               (getattr(self.cfg,'extra',{}) or {}).get('batch_token_cap'))
            for r, res in zip(chunk, self._run_safe(chunk)):
                out[r.request_id] = res
            i += len(chunk)
        return [out[r.request_id] for r in reqs]

    def _run_safe(self, batch):
        try:
            return self._generate_batch(batch)
        except Exception as exc:
            if not is_oom(exc):
                raise
            self.bump("oom")
            self._empty_cache()
            self.log_error({"kind": "oom", "batch": len(batch), "error": repr(exc)[:300],
                            "request_ids": [r.request_id for r in batch]})
            if len(batch) == 1:
                self.bump("op_errors")
                return [self.error_result(batch[0], "cuda_oom at batch size 1")]
            half = len(batch) // 2
            self.max_batch = max(1, half)
            return self._run_safe(batch[:half]) + self._run_safe(batch[half:])

    def _empty_cache(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _generate_batch(self, batch):
        """One padded batch per repetition_penalty value (normally a single group)."""
        groups = {}
        for r in batch:
            groups.setdefault(float(r.repetition_penalty), []).append(r)
        res = {}
        for rp, group in groups.items():
            for r, g in zip(group, self._generate_group(group, rp)):
                res[r.request_id] = g
        return [res[r.request_id] for r in batch]

    def _generate_group(self, group, rp):
        import torch

        enc = self.tokenizer([r.prompt for r in group], return_tensors="pt", padding=True,
                             add_special_tokens=False).to(self.device)
        plen = enc["input_ids"].shape[1]
        max_new = max(int(r.max_new_tokens) for r in group)
        stopper=None;extra={}
        if (getattr(self.cfg,'extra',{}) or {}).get('degeneration_early_stop'):
            from transformers import StoppingCriteria,StoppingCriteriaList
            from .degeneration import make_cycle_stopper
            stopper=make_cycle_stopper(torch,StoppingCriteria,group,enc['input_ids'],plen,self.pad_id,self.eos_ids,
                interval=int((self.cfg.extra or {}).get('degeneration_check_interval',16)),
                gpu=(self.cfg.extra or {}).get('degeneration_detector')=='gpu_tensor')
            extra['stopping_criteria']=StoppingCriteriaList([stopper])
        t0 = time.time()
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False, num_beams=1,
                                      repetition_penalty=rp, pad_token_id=self.pad_id,
                                      eos_token_id=sorted(self.eos_ids), temperature=None,
                                      top_p=None, top_k=None,**extra)
        secs = time.time() - t0
        flags,detector_profile=stopper.report() if stopper is not None and hasattr(stopper,'report') else (stopper.flagged if stopper else {},None)
        results = []
        for i, r in enumerate(group):
            ids, eos = postprocess_ids(out[i, plen:].tolist(), self.eos_ids, int(r.max_new_tokens), self.pad_id)
            aborted=i in flags
            if aborted:eos=False
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            n_in = int(enc["attention_mask"][i].sum().item())
            results.append(GenResult(request_id=r.request_id, text=text, eos=eos,
                                     stop_type="degeneration_abort" if aborted else ("eos" if eos else "limit"), input_tokens=n_in,
                                     output_tokens=len(ids) + (1 if eos else 0),
                                     seconds=round(secs, 3), backend=self.name))
        if getattr(self, "error_sink", None):
            lengths = [int(enc["attention_mask"][i].sum().item()) for i in range(len(group))]
            self.error_sink({"kind": "batch_profile", "seconds": round(secs, 3),
                "batch_size": len(group), "padded_prompt_tokens": plen,
                "prompt_tokens": lengths, "max_new_tokens": max_new,
                "own_budgets": [int(r.max_new_tokens) for r in group],
                "output_tokens": [r.output_tokens for r in results],
                "peak_allocated_vram":int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "peak_reserved_vram":int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
                "padded_sequence_token_allowance":len(group)*(plen+max_new),
                "degeneration_aborts":{group[i].request_id:info for i,info in flags.items()},
                "loop_detector_profile":detector_profile,
                "request_ids": [r.request_id for r in group]})
        return results

    # -- info ----------------------------------------------------------------------------------
    def peak_vram(self):
        if self.model is None:
            return None
        try:
            import torch
            if torch.cuda.is_available():
                return int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        return None

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model": self.model_path,
            "parameter_count": self.parameter_count,
            "parameter_count_source": ("counted after load: sum(p.numel() for p in model.parameters())"
                                       if self.parameter_count is not None else "not_checked (model not loaded)"),
            "device": self.device,
            "dtype": self.dtype_name,
            "gpu_name": self.gpu_name,
            "versions": self.versions,
            "peak_vram": self.peak_vram(),
            "max_batch_now": self.max_batch,
            "eos_ids": sorted(self.eos_ids),
            "decode": self.decode_config(),
        }

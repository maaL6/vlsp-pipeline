"""Replay backend: answers each request with the output recorded for the same prompt in existing
generation_trace.jsonl files (matched by sha256 of the full prompt). No model, no GPU.

Used to prove that a refactor or an offline-only change (validator, selection, assembly) keeps or
changes predictions, without a new GPU run. A prompt that is not in the traces (for example a new
recovery request after a packing change) gets a deterministic stand-in: the request payload echoed
back with EOS, counted in `stats["replay_miss"]` and listed in `misses`. Outputs produced from
stand-ins are not translations; a run with misses is only valid for equivalence checks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..core.datatypes import GenResult
from .base import Backend


class ReplayStore:
    """prompt sha256 -> recorded result fields, loaded from one or more trace files."""

    FIELDS = ("output", "eos", "stop_type", "input_tokens", "output_tokens")

    def __init__(self, trace_paths):
        self.by_sha: dict = {}
        self.conflicts = 0
        self.sources = [str(p) for p in trace_paths]
        for p in trace_paths:
            for line in Path(p).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("error") or rec.get("stop_type") == "error":
                    continue
                sha = rec["prompt_sha256"]
                val = {k: rec.get(k) for k in self.FIELDS}
                old = self.by_sha.get(sha)
                if old is None:
                    self.by_sha[sha] = val
                elif old != val:
                    self.conflicts += 1

    def lookup(self, prompt: str):
        return self.by_sha.get(hashlib.sha256(prompt.encode("utf-8")).hexdigest())


def replay_result(store: ReplayStore, req, result_cls, backend_name: str, misses: list):
    hit = store.lookup(req.prompt)
    if hit is not None:
        return result_cls(request_id=req.request_id, text=hit["output"] or "", eos=bool(hit["eos"]),
                          stop_type=hit["stop_type"] or ("eos" if hit["eos"] else "limit"),
                          input_tokens=int(hit["input_tokens"] or 0), output_tokens=int(hit["output_tokens"] or 0),
                          seconds=0.0, backend=backend_name), True
    misses.append(req.request_id)
    payload = (req.meta or {}).get("payload", "")
    return result_cls(request_id=req.request_id, text=payload, eos=True, stop_type="eos", input_tokens=0,
                      output_tokens=0, seconds=0.0, backend=backend_name), False


class ReplayBackend(Backend):
    name = "replay"
    batching = True

    def __init__(self, trace_paths, cfg=None, tok=None):
        super().__init__()
        self.cfg = cfg
        self.store = ReplayStore(trace_paths)
        self.misses: list = []
        self.stats["replay_hit"] = 0
        self.stats["replay_miss"] = 0

    def decode_config(self) -> dict:
        return {"backend": self.name, "do_sample": False,
                "repetition_penalty": float(getattr(self.cfg, "repetition_penalty", 1.05))}

    def describe(self) -> dict:
        return {"backend": self.name, "traces": self.store.sources, "recorded_prompts": len(self.store.by_sha),
                "conflicts": self.store.conflicts, "parameter_count": None,
                "parameter_count_source": "not_checked (replay, no model)", "decode": self.decode_config()}

    def generate(self, requests):
        out = []
        for r in requests:
            res, hit = replay_result(self.store, r, GenResult, self.name, self.misses)
            self.bump("replay_hit" if hit else "replay_miss")
            out.append(res)
        return out

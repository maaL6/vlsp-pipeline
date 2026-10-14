"""Deterministic scripted fake backend for tests (no model, never a translation).

Default behaviour is an echo of the unit source with marker preservation: every
"[SEG_000123] text" line comes back as "[SEG_000123] transform(text)" and EOS is reported.
Faults are scripted with rules matched against the request (meta dict, prompt, or a callable):

  drop_marker        remove one marker (arg = segment index, default 1, or 0 for one marker)
  duplicate_marker   repeat one marker line (arg = index, default 0)
  swap_markers       swap two adjacent marker lines (arg = index, default 0)
  no_eos             half of the echo, eos False, stop_type "limit" (budget hit)
  empty              "" with EOS
  loop               echo + a repeated phrase, eos False (arg = phrase)
  text_before_marker free text before the first marker (unassigned_text)
  leak_marker        echo + an internal-looking marker (arg, default "[SEG_000999]")
  replace            exact text (arg), with EOS
  error              operational error result (stop_type "error")
  callable           fault(req, segments, echo) -> str | (str, eos) | GenResult

The backend records every call (meta, prompt, fault), the batches it received and the maximum
number of in-flight requests globally and per document (scheduling tests).
"""
from __future__ import annotations

import re
import threading
import time
from collections import Counter
from dataclasses import dataclass

from .base import Backend
from ..core.datatypes import GenResult
from ..steps.s04_prompt import ASSISTANT, BOS, USER

MARKER_RE = re.compile(r"\[(SEG[0-9A-F]{0,4})_(\d{6})\]")

FAULTS = ("echo", "drop_marker", "duplicate_marker", "swap_markers", "no_eos", "empty", "loop",
          "text_before_marker", "leak_marker", "replace", "error")

_ANCHORS = ("[Source Text]\n", "any additional explanation:\n\n",
            "pay close attention to their placement.\n\n")

@dataclass
class Rule:
    fault: Union[str, Callable]
    match: Union[dict, Callable, None] = None   # None = every request
    times: Optional[int] = None                 # max number of applications (None = unlimited)
    arg: Any = None
    used: int = 0

    def matches(self, req) -> bool:
        if self.times is not None and self.used >= self.times:
            return False
        if self.match is None:
            return True
        if callable(self.match):
            return bool(self.match(req))
        meta = req.meta or {}
        for k, v in self.match.items():
            if k == "prompt_contains":
                if v not in req.prompt:
                    return False
            elif k == "node_id":
                if v not in (meta.get("node_ids") or []):
                    return False
            elif k == "min_markers":
                if len(meta.get("markers") or []) < v:
                    return False
            elif meta.get(k) != v:
                return False
        return True

    @property
    def label(self) -> str:
        return self.fault if isinstance(self.fault, str) else getattr(self.fault, "__name__", "callable")

def source_of(prompt: str) -> str:
    """Source text of a rendered HY prompt (what the model is asked to translate)."""
    user = prompt
    if user.startswith(BOS):
        user = user[len(BOS):]
    if user.startswith(USER):
        user = user[len(USER):]
    if user.endswith(ASSISTANT):
        user = user[: -len(ASSISTANT)]
    for anchor in _ANCHORS:
        if anchor in user:
            return user.split(anchor, 1)[1]
    return user

def split_segments(src: str):
    """[(marker or None, text)]: text before the first marker gets marker None."""
    ms = list(MARKER_RE.finditer(src))
    if not ms:
        return [(None, src.strip())]
    segs = []
    pre = src[: ms[0].start()].strip()
    if pre:
        segs.append((None, pre))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(src)
        segs.append((m.group(0), src[m.end():end].strip()))
    return segs

def render_segments(segs, transform=None) -> str:
    f = transform or (lambda s: s)
    return "\n".join(f"{m} {f(t)}" if m else f(t) for m, t in segs)

class MockBackend(Backend):
    name = "mock"

    def __init__(self, rules=(), tok=None, transform=None, delay: float = 0.0, batching: bool = False,
                 label: str = "echo"):
        super().__init__()
        self.rules = [r if isinstance(r, Rule) else Rule(**r) for r in rules]
        self.tok = tok
        self.transform = transform
        self.delay = float(delay)
        self.batching = bool(batching)
        self.label = label
        self.calls: list = []
        self.batches: list = []
        self.inflight = 0
        self.max_inflight = 0
        self._by_doc = Counter()
        self.max_inflight_by_doc: dict = {}

    def add_rule(self, fault, match=None, times=None, arg=None) -> Rule:
        r = Rule(fault=fault, match=match, times=times, arg=arg)
        self.rules.append(r)
        return r

    # -- interface -----------------------------------------------------------------------------
    def decode_config(self) -> dict:
        return {"backend": self.name, "label": self.label, "do_sample": False,
                "rules": [r.label for r in self.rules]}

    def describe(self) -> dict:
        return {"backend": self.name, "label": self.label,
                "note": "deterministic echo fake for tests, NOT a translation model",
                "parameter_count": 0, "parameter_count_source": "mock (no model)",
                "rules": [{"fault": r.label, "times": r.times, "used": r.used} for r in self.rules],
                "device": "cpu", "dtype": "n/a", "peak_vram": None}

    def generate(self, requests):
        reqs = list(requests)
        docs = [str((r.meta or {}).get("doc_id", "")) for r in reqs]
        with self._lock:
            self.batches.append([r.request_id for r in reqs])
            self.inflight += len(reqs)
            self.max_inflight = max(self.max_inflight, self.inflight)
            for d in docs:
                self._by_doc[d] += 1
                self.max_inflight_by_doc[d] = max(self.max_inflight_by_doc.get(d, 0), self._by_doc[d])
        try:
            if self.delay:
                time.sleep(self.delay)
            return [self._one(r) for r in reqs]
        finally:
            with self._lock:
                self.inflight -= len(reqs)
                for d in docs:
                    self._by_doc[d] -= 1

    # -- internals -----------------------------------------------------------------------------
    def _pick(self, req) -> Optional[Rule]:
        with self._lock:
            for r in self.rules:
                if r.matches(req):
                    r.used += 1
                    return r
        return None

    def _count(self, text: str) -> int:
        return self.tok.count(text) if self.tok is not None else len(text.split())

    def _one(self, req) -> GenResult:
        t0 = time.time()
        meta = req.meta or {}
        src = meta.get("payload")
        if src is None:
            src = source_of(req.prompt)
        segs = split_segments(src)
        echo = render_segments(segs, self.transform)
        rule = self._pick(req)
        fault = rule.fault if rule else "echo"
        label = rule.label if rule else "echo"
        with self._lock:
            self.calls.append({"request_id": req.request_id, "meta": dict(meta), "prompt": req.prompt,
                               "max_new_tokens": req.max_new_tokens, "fault": label,
                               "thread": threading.current_thread().name, "t": t0})
        if fault == "error":
            self.bump("op_errors")
            self.log_error({"request_id": req.request_id, "kind": "mock", "error": "scripted error"})
            return self.error_result(req, "mock scripted operational error")
        text, eos = self._apply(fault, rule, segs, echo, req)
        if isinstance(text, GenResult):
            return text
        n_out = self._count(text)
        if eos and self.tok is not None and n_out > req.max_new_tokens:  # budget is a hard limit
            text, eos, n_out = self.tok.truncate_tokens(text, req.max_new_tokens), False, req.max_new_tokens
        if not eos:
            n_out = max(n_out, int(req.max_new_tokens))
        else:
            n_out += 1  # EOS token counted like llama.cpp tokens_predicted
        return GenResult(request_id=req.request_id, text=text, eos=eos, stop_type="eos" if eos else "limit",
                         input_tokens=self._count(req.prompt), output_tokens=n_out,
                         seconds=round(time.time() - t0, 4), backend=self.name)

    def _apply(self, fault, rule, segs, echo, req):
        arg = rule.arg if rule else None
        marked = [i for i, (m, _) in enumerate(segs) if m]
        f = self.transform or (lambda s: s)
        if callable(fault):
            out = fault(req, segs, echo)
            if isinstance(out, GenResult):
                return out, True
            if isinstance(out, tuple):
                return out
            return out, True
        if fault == "echo":
            return echo, True
        if fault == "replace":
            return str(arg or ""), True
        if fault == "empty":
            return "", True
        if fault == "no_eos":
            return echo[: max(1, len(echo) // 2)], False
        if fault == "loop":
            phrase = arg or "lặp lại"
            return echo + " " + " ".join([phrase] * 200), False
        if fault == "text_before_marker":
            return (arg or "Bản dịch như sau:") + "\n" + echo, True
        if fault == "leak_marker":
            return echo + " " + (arg or "[SEG_000999]"), True
        if fault in ("drop_marker", "duplicate_marker", "swap_markers"):
            if not marked:
                return echo, True
            lines = [(m, f(t)) for m, t in segs]
            if fault == "drop_marker":
                k = arg if arg is not None else (1 if len(marked) > 1 else 0)
                i = marked[min(k, len(marked) - 1)]
                lines[i] = (None, lines[i][1])
            elif fault == "duplicate_marker":
                i = marked[min(arg or 0, len(marked) - 1)]
                lines.insert(i + 1, lines[i])
            else:
                k = min(arg or 0, max(0, len(marked) - 2))
                if len(marked) >= 2:
                    a, b = marked[k], marked[k + 1]
                    lines[a], lines[b] = lines[b], lines[a]
            return "\n".join(f"{m} {t}" if m else t for m, t in lines), True
        raise ValueError(f"unknown mock fault {fault!r}; expected one of {FAULTS} or a callable")

from __future__ import annotations

from collections import Counter, deque

def _parallel(gens):
    """Run unit sub-generators side by side: one combined yield per round."""
    results = [None] * len(gens)
    pending = {}

    def step(i, value):
        g = gens[i]
        while True:
            try:
                reqs = g.send(value)
            except StopIteration as e:
                results[i] = e.value
                return
            if reqs:
                pending[i] = list(reqs)
                return
            value = []

    for i in range(len(gens)):
        step(i, None)
    while pending:
        order = sorted(pending)
        combined = [r for i in order for r in pending[i]]
        got = yield combined
        by_id = {r.request_id: x for r, x in zip(combined, got)}
        for i in order:
            reqs = pending.pop(i)
            step(i, [by_id[r.request_id] for r in reqs])
    return results

# -- scheduler ---------------------------------------------------------------------------------------
class _Runner:
    """Drives one document generator (yield list[GenRequest] -> receive list[GenResult])."""

    def __init__(self, job: _DocJob, limit: Optional[int]):
        self.job, self.limit = job, limit
        self.gen = job.flow()
        self.waiting = deque()
        self.expected: list = []
        self.got: dict = {}
        self.inflight = 0
        self.done = False

    def start(self):
        self._advance(None)

    def _advance(self, value):
        while True:
            try:
                reqs = self.gen.send(value)
            except StopIteration:
                self.done = True
                return
            except Exception as exc:  # any failure of this document only
                self.done = True
                self.job.fail(exc)
                return
            if reqs:
                self.expected = [r.request_id for r in reqs]
                self.got = {}
                self.waiting = deque(reqs)
                return
            value = []

    def ready(self) -> bool:
        return (not self.done and bool(self.waiting)
                and (self.limit is None or self.inflight < self.limit))

    def take(self) -> GenRequest:
        self.inflight += 1
        return self.waiting.popleft()

    def deliver(self, req, res):
        self.inflight -= 1
        self.got[req.request_id] = res
        if not self.waiting and self.inflight == 0 and len(self.got) == len(self.expected):
            self._advance([self.got[i] for i in self.expected])

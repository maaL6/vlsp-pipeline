from __future__ import annotations

import statistics

DOC_STATUSES = ("completed", "completed_with_unresolved", "failed")

SOURCE_KINDS = ("title_path", "definition", "reference")

# -- small helpers -----------------------------------------------------------------------------------
def _code(c) -> str:
    return str(c).split(":", 1)[0]

def _dedupe(xs) -> list:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _stats(xs) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "sum": int(sum(xs)), "min": min(xs), "median": statistics.median(xs), "max": max(xs)}

"""Split from lam_docmt/units.py (statements verbatim; tools/split_from_lam_docmt.py)."""
from __future__ import annotations

from .s04_prompt import build_prompt, model_family

# --------------------------------------------------------------------------------------- budget
def target_budget(source_tokens: int) -> int:
    """max(256, ceil(2.5 * source_tokens) + 128)."""
    return max(256, -(-5 * int(source_tokens) // 2) + 128)

def output_budget(source_tokens, prompt_tokens, cfg) -> int:
    """min(max_sequence - prompt_tokens - margin, max(256, ceil(2.5*source_tokens)+128)).
    Can be < target (or <= 0) when the prompt is too long: check `budget_fits` first."""
    return min(cfg.max_sequence - int(prompt_tokens) - cfg.margin, target_budget(source_tokens))

def budget_fits(source_tokens, prompt_tokens, cfg) -> bool:
    """True when prompt + full target budget + margin fits in max_sequence."""
    return int(prompt_tokens) + target_budget(source_tokens) + cfg.margin <= cfg.max_sequence


# ---- request fitting (moved from engine/doc_job.py _DocJob._fits_alone/_build) ---------------------
def fits_alone(unit, tgt_lang: str, tok, cfg, output_budget=output_budget) -> bool:
    """The unit's source with no evidence fits: prompt + full budget + margin <= max_sequence."""

    if model_family(cfg) == "index":
        from .s04_prompt_index import build_prompt as build_index_prompt
        prompt = build_index_prompt(unit.payload, tgt_lang, has_markers=bool(unit.markers))
    else:
        prompt = build_prompt(unit.payload, tgt_lang, cfg.prompt_style, has_markers=bool(unit.markers))
    want = output_budget(unit.source_tokens, 0, cfg)
    return tok.count(prompt) + want + cfg.margin <= cfg.max_sequence


def recovery_source_tokens(unit, tok, cfg):
    """(ok, tokens): a recovery request must carry <= recovery_cap source tokens."""
    n = tok.count(unit.payload)
    return n <= cfg.recovery_cap, n


def fit_prompt(unit, items, render, tok, cfg, output_budget=output_budget):
    """Drop evidence items from the end (lowest priority) until prompt + budget + margin fits.
    Returns (prompt, template, prompt_tokens, max_new_tokens, kept_items, dropped_items), or None
    when even the prompt without evidence does not fit."""
    items = list(items)
    dropped = []
    want = output_budget(unit.source_tokens, 0, cfg)
    while True:
        prompt, template = render(unit, items)
        ptoks = tok.count(prompt)
        if ptoks + want + cfg.margin <= cfg.max_sequence:
            break
        if not items:
            return None
        dropped.append(items.pop())  # lowest priority last
    budget = int(output_budget(unit.source_tokens, ptoks, cfg))
    return prompt, template, ptoks, budget, items, dropped

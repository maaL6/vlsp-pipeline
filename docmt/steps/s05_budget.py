from __future__ import annotations

from .s04_prompt import build_prompt

# --------------------------------------------------------------------------------------- budget
def target_budget(source_tokens: int, cfg=None, tgt_lang=None, protected=False, full=False) -> int:
    """Directional prose allowance; legacy allowance for protected kinds and recovery."""
    if cfg is not None and (cfg.extra or {}).get('directional_output_budget') and tgt_lang=='en' and not protected and not full:
        return max(256,-(-5*int(source_tokens)//4)+128)
    return max(256, -(-5 * int(source_tokens) // 2) + 128)

def sequence_limit(source_tokens, prompt_tokens, cfg, want=None):
    """8k normally; 16k ceiling only when a whole sentence needs its full output allowance."""
    normal=int(cfg.max_sequence)
    need=int(prompt_tokens)+(target_budget(source_tokens) if want is None else want)+cfg.margin
    ceiling=int((cfg.extra or {}).get("sentence_sequence_ceiling",normal))
    return normal if need<=normal else max(normal,ceiling)

def output_budget(source_tokens, prompt_tokens, cfg) -> int:
    """min(max_sequence - prompt_tokens - margin, max(256, ceil(2.5*source_tokens)+128)).
    Can be < target (or <= 0) when the prompt is too long: check `budget_fits` first."""
    return min(sequence_limit(source_tokens,prompt_tokens,cfg) - int(prompt_tokens) - cfg.margin, target_budget(source_tokens))

def budget_fits(source_tokens, prompt_tokens, cfg) -> bool:
    """True when prompt + full target budget + margin fits in max_sequence."""
    return int(prompt_tokens) + target_budget(source_tokens) + cfg.margin <= sequence_limit(source_tokens,prompt_tokens,cfg)


# ---- request fitting (moved from engine/doc_job.py _DocJob._fits_alone/_build) ---------------------
def fits_alone(unit, tgt_lang: str, tok, cfg, output_budget=output_budget) -> bool:
    """The unit's source with no evidence fits: prompt + full budget + margin <= max_sequence."""

    prompt = build_prompt(unit.payload, tgt_lang, cfg.prompt_style, has_markers=bool(unit.markers))
    want = target_budget(unit.source_tokens,cfg,tgt_lang,getattr(unit,'kind','text') in ('table','signature'))
    ptoks=tok.count(prompt)
    return ptoks + want + cfg.margin <= sequence_limit(unit.source_tokens,ptoks,cfg,want)


def recovery_source_tokens(unit, tok, cfg):
    """(ok, tokens): a recovery request must carry <= recovery_cap source tokens."""
    n = tok.count(unit.payload)
    return n <= cfg.recovery_cap or bool((cfg.extra or {}).get("sentence_only")), n


def fit_prompt(unit, items, render, tok, cfg, output_budget=output_budget, *, tgt_lang=None, protected=False, full=False):
    """Drop evidence items from the end (lowest priority) until prompt + budget + margin fits.
    Returns (prompt, template, prompt_tokens, max_new_tokens, kept_items, dropped_items), or None
    when even the prompt without evidence does not fit."""
    items = list(items)
    dropped = []
    directional=bool((cfg.extra or {}).get('directional_output_budget'))
    want = target_budget(unit.source_tokens,cfg,tgt_lang,protected or getattr(unit,'kind','text') in ('table','signature'),full)
    while True:
        prompt, template = render(unit, items)
        ptoks = tok.count(prompt)
        if ptoks + want + cfg.margin <= sequence_limit(unit.source_tokens,ptoks,cfg,want):
            break
        if not items:
            return None
        dropped.append(items.pop())  # lowest priority last
    budget = want if directional else int(output_budget(unit.source_tokens, ptoks, cfg))
    return prompt, template, ptoks, budget, items, dropped

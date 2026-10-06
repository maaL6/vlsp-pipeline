"""HY A1 prompts. The rendered string equals hy_a1_tokenizer/chat_template.jinja applied to one user
turn with add_generation_prompt=True. A1 was SFT-ed on HY_DEFAULT (about 1/3 of rows on HY_TERMS);
HY_DELIMITER and HY_BACKGROUND come from the Hy-MT2 model card and were not part of A1 SFT."""
# Split from lam_docmt/prompts.py, lam_docmt/runtime.py (statements verbatim; tools/split_from_lam_docmt.py).
from __future__ import annotations

import re

NAMES = {"en": "English", "vi": "Vietnamese"}

BOS = "<｜hy_begin▁of▁sentence｜>"

USER = "<｜hy_User｜>"

ASSISTANT = "<｜hy_Assistant｜>"

EOS = "<｜hy_place▁holder▁no▁2｜>"          # id 120020 (generation_config eos_token_id)

END_NO_GEN = "<｜hy_place▁holder▁no▁8｜>"   # appended by the template when add_generation_prompt=False

HY_DEFAULT = ("Translate the following text into {tgt}. Note that you should only output the "
              "translated result without any additional explanation:\n\n{src}")

HY_TERMS = ("Reference the following translations:\n{refs}\n\n"
            "Translate the following text into {tgt}. Note that you must ONLY output the "
            "translated result without any additional explanation:\n\n{src}")

HY_DELIMITER = ("Please accurately translate the following text into {tgt}.\n"
                "You must retain the exact same number of delimiters in the translation. Strictly do "
                "not omit, escape, or translate these symbols, and pay close attention to their "
                "placement.\n\n{src}")

HY_BACKGROUND = ("[Background Information]\n{bg}\n\n"
                 "Please translate the following text into {tgt}, taking the provided background "
                 "information into consideration.{extra}\n\n[Source Text]\n{src}")

BACKGROUND_ONLY_SOURCE = (" Only output the translation of [Source Text]; do not translate or repeat "
                          "the background information.")

DELIMITER_NOTE = (" You must retain the exact same number of delimiters in the translation. Strictly "
                  "do not omit, escape, or translate these symbols, and pay close attention to their "
                  "placement.")

PROMPT_STYLES = ("default", "delimiter")

def render_chat(user: str) -> str:
    return f"{BOS}{USER}{user}{ASSISTANT}"

def build_user(src: str, tgt_lang: str, style: str = "default", background: str = "",
               terms=(), has_markers: bool = False) -> str:
    """User content for one request.

    - no background, no terms: HY_DEFAULT (style "default") or HY_DELIMITER (style "delimiter",
      only when the payload carries markers; single-node payloads always use HY_DEFAULT).
    - terms only: HY_TERMS (A1-native terminology format), "<src> translates to <tgt>" lines.
    - background: HY_BACKGROUND; terms are listed inside the background block; with markers and
      style "delimiter" the delimiter sentence is appended to the instruction.
    """
    tgt = NAMES[tgt_lang]
    refs = "\n".join(f"{a} translates to {b}" for a, b in terms)
    if background:
        bg = background
        if refs:
            bg = bg + "\nReference translations:\n" + refs
        extra = BACKGROUND_ONLY_SOURCE + (DELIMITER_NOTE if (has_markers and style == "delimiter") else "")
        return HY_BACKGROUND.format(bg=bg, tgt=tgt, extra=extra, src=src)
    if refs:
        return HY_TERMS.format(refs=refs, tgt=tgt, src=src)
    if has_markers and style == "delimiter":
        return HY_DELIMITER.format(tgt=tgt, src=src)
    return HY_DEFAULT.format(tgt=tgt, src=src)

def build_prompt(src: str, tgt_lang: str, style: str = "default", background: str = "",
                 terms=(), has_markers: bool = False) -> str:
    return render_chat(build_user(src, tgt_lang, style, background, terms, has_markers))

def _prompt_template(cfg, background: str, terms, has_markers: bool) -> str:
    if background:
        t = "HY_BACKGROUND" + ("+delimiter" if has_markers and cfg.prompt_style == "delimiter" else "")
    elif terms:
        t = "HY_TERMS"
    elif has_markers and cfg.prompt_style == "delimiter":
        t = "HY_DELIMITER"
    else:
        t = "HY_DEFAULT"
    return f"{cfg.prompt_style}:{t}"

_TERM_LINE = re.compile(r"^\s*(.+?)\s+(?:translates to|=|->|→)\s+(.+?)\s*$")

def term_pair(item, src_lang: str, tgt_lang: str):
    """(source term, target term) of a glossary EvidenceItem, or None."""
    ref = item.ref or {}
    for a, b in ((src_lang, tgt_lang), ("src", "tgt"), ("source", "target"), ("term", "translation")):
        if isinstance(ref.get(a), str) and isinstance(ref.get(b), str) and ref[a].strip() and ref[b].strip():
            return ref[a].strip(), ref[b].strip()
    m = _TERM_LINE.match(item.text or "")
    return (m.group(1), m.group(2)) if m else None


# ---- unit prompt (moved from engine/doc_job.py _DocJob._render) ------------------------------------
def render_unit_prompt(unit, items, src_lang: str, tgt_lang: str, cfg, render_background):
    """(prompt, template label) for one unit. Glossary evidence that parses as a term pair goes to
    the HY_TERMS / background term list; every other evidence item goes to the background block."""
    terms, bg = [], []
    for it in items:
        pair = term_pair(it, src_lang, tgt_lang) if it.kind == "glossary" else None
        if pair:
            terms.append(pair)
        else:
            bg.append(it)
    background = render_background(bg) if bg else ""
    has_markers = bool(unit.markers)
    if model_family(cfg) == "index":
        from . import s04_prompt_index as idx
        prompt = idx.build_prompt(unit.payload, tgt_lang, context=background, terms=terms, has_markers=has_markers)
        return prompt, idx.template_label(background, terms, has_markers)
    prompt = build_prompt(unit.payload, tgt_lang, cfg.prompt_style, background=background,
                          terms=terms, has_markers=has_markers)
    return prompt, _prompt_template(cfg, background, terms, has_markers)


def model_family(cfg) -> str:
    """"hy" (default, A1 prompts above) or "index" (s04_prompt_index); set in cfg.extra so the
    RunConfig fields, and therefore the config hash of existing HY runs, do not change."""
    return str((getattr(cfg, "extra", None) or {}).get("model_family", "hy"))

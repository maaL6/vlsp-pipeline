"""Index-Translate-2B native prompts (Trang, 06/10/2026).

Same strings as Trang's `index_native.py`, used to build the legal-SFT pilot data (Linh data_delivery_v2)
and to train the adapter, so the inference prompt equals the training prompt:

- no evidence/markers: the Index client's standard translation prompt (Chinese instruction, explicit
  source/target language names);
- evidence, terms or markers: the Index instTrans format (【源文】 + 【约束要求】); structure and markers are
  【硬性要求】 (hard), source context and glossary are 【注意】 (soft).

The rendered string equals the model's chat template for one user turn with add_generation_prompt=True
and enable_thinking=False (ends with an empty think block).
"""
from __future__ import annotations

LANG_NAMES = {"en": "英语", "vi": "越南语"}
OTHER = {"en": "vi", "vi": "en"}
GENRE = "法律文本"
MAX_GLOSSARY_TERMS = 8

HARD_STRUCTURE = "【硬性要求】保留源文的 Markdown 结构、条/款/项编号以及所有数字和金额原样不变"
HARD_MARKERS = "【硬性要求】保留源文中所有段落标记（如 [SEG_000001]）原样不变，数量与顺序与源文一致"
SOFT_GLOSSARY = "【注意】术语参考（仅在语境相符时采用）: "
SOFT_CONTEXT = "【注意】以下是同一文件中的前文原文，仅用于理解语境，不要翻译或输出：\n"

CHAT_PREFIX = "<|im_start|>user\n"
CHAT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_user(src: str, src_lang: str, tgt_lang: str, context: str = "", terms=(), has_markers: bool = False) -> str:
    src_name, tgt_name = LANG_NAMES[src_lang], LANG_NAMES[tgt_lang]
    glossary = [f"{a}→{b}" for a, b in terms][:MAX_GLOSSARY_TERMS]
    if not (context or glossary or has_markers):
        return f"请将以下{src_name}文本翻译为{tgt_name}，直接输出翻译结果，不要进行任何解释。\n\n{src.strip()}"
    constraints = [HARD_STRUCTURE]
    if has_markers:
        constraints.append(HARD_MARKERS)
    if glossary:
        constraints.append(SOFT_GLOSSARY + "、".join(glossary))
    if context:
        constraints.append(f'{SOFT_CONTEXT}"""\n{context.strip()}\n"""')
    req = "\n".join(f"{i}. {c}" for i, c in enumerate(constraints, 1))
    return (f"请将以下{src_name}{GENRE}翻译成{tgt_name}，并且严格遵循所有约束要求。\n\n"
            f"【源文】\n{src.strip()}\n\n【约束要求】\n{req}\n\n只输出译文，不要有任何额外说明。")


def render_chat(user: str) -> str:
    return f"{CHAT_PREFIX}{user}{CHAT_SUFFIX}"


def build_prompt(src: str, tgt_lang: str, context: str = "", terms=(), has_markers: bool = False) -> str:
    return render_chat(build_user(src, OTHER[tgt_lang], tgt_lang, context, terms, has_markers))


def template_label(context: str, terms, has_markers: bool) -> str:
    return "index:" + ("instTrans" if (context or terms or has_markers) else "default")


def strip_thinking(text: str) -> str:
    """Drop a think block if the model emits one (same rule as the Index client)."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1].lstrip("\n")
    return text

from __future__ import annotations

import re

from .common import _nfc, _ws

# ============================================================================ text heuristics
_VI_CHARS = set("ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ")

_WORD = re.compile(r"[^\W\d_]+")

def _vi_word_ratio(text):
    words = _WORD.findall(_nfc(text))
    if not words:
        return 0.0, 0
    vi = sum(1 for w in words if any(c.lower() in _VI_CHARS for c in w))
    return vi / len(words), len(words)

def _untranslated(payload, seg, tgt_lang):
    """'identical' | 'wrong_script' | None (heuristic)."""
    p, s = _ws(_nfc(payload)).casefold(), _ws(_nfc(seg)).casefold()
    words = _WORD.findall(_nfc(payload))
    if p == s and len([w for w in words if len(w) >= 2]) >= 3 \
            and not all(w[:1].isupper() and w[1:].islower() for w in words):
        return "identical"
    ratio, n = _vi_word_ratio(seg)
    if n >= 6 and ((tgt_lang == "vi" and ratio < 0.1) or (tgt_lang == "en" and ratio > 0.5)):
        return "wrong_script"
    return None

_NEG_EXCLUDE_EN = re.compile(r"\b(?:no|not)\s+(?:later|earlier|sooner|more|less|fewer|greater|higher|lower|"
                             r"longer|shorter)\s+than\b|\bnot\s+(?:to\s+)?exceed(?:ing)?\b", re.I)

_NEG_EN = re.compile(r"\b(?:not|never|neither|nor|none|nothing|nobody|cannot|without)\b|n't\b|"
                     r"\bno\b(?!\.?\s*\d)|\bnon-|\bfail(?:s|ed|ure)?\s+to\b", re.I)

_NEG_EXCLUDE_VI = re.compile(r"\bkhông\s+(?:quá|muộn hơn|sớm hơn|ít hơn|thấp hơn|cao hơn|vượt quá|dưới|"
                             r"nhiều hơn|chậm hơn|lâu hơn)\b|\b(?:hàng|phòng)\s+không\b|"
                             r"\bkhông\s+(?:khí|gian|quân|lưu|phận|ảnh)\b", re.I)

_NEG_VI = re.compile(r"\b(?:không|chưa|chẳng|chả|đừng|chớ)\b", re.I)

def negation_count(text, lang):
    t = _nfc(text)
    if lang == "en":
        return len(_NEG_EN.findall(_NEG_EXCLUDE_EN.sub(" ", t)))
    return len(_NEG_VI.findall(_NEG_EXCLUDE_VI.sub(" ", t)))

# strict = counted on the SOURCE side, lenient = counted on the OUTPUT side, so that only a
# clear loss of an obligation/permission marker is flagged ("shall"/"được" are too ambiguous
# to count on the source side).
_MODAL = {
    ("en", "strict"): re.compile(r"\b(?:must|(?:is|are|be)\s+required\s+to|obliged\s+to|obligated\s+to|"
                                 r"ha(?:s|ve)\s+to|may(?!\s*\d)|entitled\s+to|permitted\s+to|allowed\s+to|"
                                 r"ha(?:s|ve)\s+the\s+rights?\s+to|can)\b", re.I),
    ("en", "lenient"): re.compile(r"\b(?:must|shall|required\s+to|obliged\s+to|obligated\s+to|ha(?:s|ve|d)\s+to|"
                                  r"responsible\s+for|needs?\s+to|duty\s+to|obligations?\s+to|liable|"
                                  r"may(?!\s*\d)|can|could|entitled|permitted|allowed|rights?\s+to|"
                                  r"eligible|authori[sz]ed\s+to|able\s+to)\b", re.I),
    ("vi", "strict"): re.compile(r"\b(?:phải|bắt buộc|có nghĩa vụ|có thể|có quyền|được phép|được quyền)\b", re.I),
    ("vi", "lenient"): re.compile(r"\b(?:phải|bắt buộc|buộc|có nghĩa vụ|có trách nhiệm|cần(?!\s+thiết)|"
                                  r"có thể|có quyền|được)\b", re.I),
}

def modal_count(text, lang, mode):
    return len(_MODAL[(lang, mode)].findall(_nfc(text)))

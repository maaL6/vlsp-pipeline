"""Chuẩn hoá văn bản cho pipeline dịch EN<->VI.

  normalize(text)                  : đúng đặc tả data_release_v2/README.md, dùng cho nguồn lúc inference
                                     (bắt buộc, để khớp với lúc train) và làm bước đầu của hậu xử lý.
  postprocess(hyp, src, tgt_lang)  : normalize() + sửa số bị copy nguyên định dạng tiếng Anh sang bản dịch tiếng Việt.

Không bao giờ áp lên reference khi chấm điểm.

  python text_normalize.py < in.txt > out.txt     # chạy normalize() cho stdin
"""
import re
import sys
import unicodedata as ud
from functools import lru_cache

# --- dấu thanh -------------------------------------------------------------------------------
# NFD: huyền, sắc, ngã, hỏi, nặng
TONES = {"̀", "́", "̃", "̉", "̣"}
VOWELS = set("aăâeêioôơuưy")
MARKED = set("ăâêôơư")  # nguyên âm có mũ/móc/trăng: luôn nhận dấu thanh
ONSETS = sorted(["ngh", "ng", "nh", "ch", "gh", "gi", "kh", "ph", "qu", "th", "tr",
                 "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s", "t", "v", "x", ""],
                key=len, reverse=True)
CODAS = {"", "c", "ch", "m", "n", "ng", "nh", "p", "t"}
# vần nguyên âm hợp lệ; từ gõ hỏng (ngưòi, cứư, iý) không khớp nên được giữ nguyên
NUCLEI = set("aăâeêioôơuưy") | {
    "ai", "ao", "au", "ay", "âu", "ây", "eo", "êu", "ia", "iê", "iu", "oa", "oă", "oe", "oi", "oo", "ôi", "ơi",
    "ua", "uâ", "uê", "ui", "uô", "uơ", "uy", "ưa", "ưi", "ươ", "ưu", "yê",
    "iêu", "oai", "oay", "oeo", "uây", "uôi", "uya", "uyê", "uyu", "ươi", "ươu", "yêu"}


def _strip_tones(word):
    """Trả về (chữ không dấu thanh ở dạng NFC, danh sách dấu thanh)."""
    tones = [c for c in ud.normalize("NFD", word) if c in TONES]
    base = ud.normalize("NFC", "".join(c for c in ud.normalize("NFD", word) if c not in TONES))
    return base, tones


def _parse(base):
    """Tách âm tiết (chữ thường, không dấu thanh) thành (âm đầu, vần nguyên âm, âm cuối) hoặc None."""
    for on in ONSETS:
        if not base.startswith(on):
            continue
        rest = base[len(on):]
        if on == "gi" and (not rest or rest[0] not in VOWELS):  # gì, gìn: i là nguyên âm
            on, rest = "g", base[1:]
        i = 0
        while i < len(rest) and rest[i] in VOWELS:
            i += 1
        nucleus, coda = rest[:i], rest[i:]
        if nucleus in NUCLEI and coda in CODAS:
            return on, nucleus, coda
    return None


def _tone_index(nucleus, coda):
    """Vị trí nhận dấu thanh trong vần, theo kiểu cũ (hòa, thủy, khỏe)."""
    marked = [i for i, c in enumerate(nucleus) if c in MARKED]
    if marked:
        return marked[-1]  # ươ -> ơ, uyê -> ê
    if coda or len(nucleus) == 3:
        return len(nucleus) - 1 if coda else 1  # hoàng, tuyến; oái, khuỷu
    return 0  # 1 nguyên âm, hoặc 2 nguyên âm không âm cuối: hòa, mùa, tùy


def _is_syllable(word):
    base, tones = _strip_tones(word)
    return len(tones) <= 1 and _parse(base.lower()) is not None


@lru_cache(maxsize=200_000)
def fix_tone(word):
    """Đặt lại dấu thanh của một âm tiết tiếng Việt; từ không phải âm tiết Việt giữ nguyên."""
    base, tones = _strip_tones(word)
    if len(tones) != 1:
        return word
    parsed = _parse(base.lower())
    if parsed is None:
        return word
    on, nucleus, coda = parsed
    pos = len(on) + _tone_index(nucleus, coda)
    out = base[:pos + 1] + tones[0] + base[pos + 1:]
    return ud.normalize("NFC", out)


_WORD = re.compile(r"(?:[^\W\d_]|[\u0300-\u036f])+")  # gồm cả dấu tổ hợp rời: hoăc\u0323

# --- ký tự lạ --------------------------------------------------------------------------------
HOMOGLYPHS = str.maketrans({"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
                            "і": "i", "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Х": "X",
                            "Ð": "Đ"})
_CYR = re.compile(r"[Ѐ-ӿ]")
_INVIS = "​‌‍‎‏⁠﻿"
_INVIS_RE = re.compile(f"(\\w?)([{_INVIS}­]+)(\\w?)")


def _fix_homoglyphs(text):
    # chỉ đổi chữ Kirin lẻ trong từ Latin; để nguyên từ Kirin thật
    def repl(m):
        w = m.group(0)
        return w.translate(HOMOGLYPHS) if len(_CYR.findall(w)) < len(w) or len(w) == 1 else w
    return _WORD.sub(repl, text).replace("Ð", "Đ")


def _fix_invisible(text):
    def repl(m):
        a, inv, b = m.groups()
        if a and b and inv != "­":
            # ký tự vô hình thay cho dấu cách giữa hai âm tiết: ý‎kiến -> ý kiến
            left = re.search(r"[^\W\d_]+$", text[:m.start(2)])
            right = re.match(r"[^\W\d_]+", text[m.end(2):])
            if left and right and _is_syllable(left.group(0)) and _is_syllable(right.group(0)) \
                    and not _is_syllable(left.group(0) + right.group(0)):
                return a + " " + b
        return a + b
    return _INVIS_RE.sub(repl, text)


def normalize(text):
    text = ud.normalize("NFC", text)
    text = _fix_homoglyphs(text)
    text = _fix_invisible(text)
    text = re.sub(r"(?<=[^\W\d_])‘(?=[^\W\d_])", "’", text)  # People‘s -> People’s
    return _WORD.sub(lambda m: fix_tone(m.group(0)), text)


# --- hậu xử lý bản dịch ----------------------------------------------------------------------
_EN_THOUSANDS = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,]|\.\d)")


def postprocess(hyp, src=None, tgt_lang="vi"):
    """Hậu xử lý bản dịch. Với en->vi: `1,234,567` -> `1.234.567` (reference dùng dấu chấm hàng nghìn).

    Chỉ đổi số (a) có dấu phẩy hàng nghìn đúng 3 chữ số, không bắt đầu bằng 0, và (b) có nguyên văn trong
    nguồn tiếng Anh, tức là model đã copy mà không đổi định dạng. Nhờ (b) mà thập phân Việt `1,500`
    (= 1,5) model tự viết không bị đụng. Không đổi thập phân `2.5` vì dễ nhầm với `Điều 2.1`.
    """
    hyp = normalize(hyp)
    if tgt_lang == "vi" and src:
        copied = {m for m in _EN_THOUSANDS.findall(src) if not m.startswith("0")}
        hyp = _EN_THOUSANDS.sub(lambda m: m.group(0).replace(",", ".") if m.group(0) in copied
                                else m.group(0), hyp)
    return hyp


if __name__ == "__main__":
    sys.stdout.write(normalize(sys.stdin.read()))

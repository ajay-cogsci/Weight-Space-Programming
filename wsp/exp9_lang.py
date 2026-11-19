"""exp9's grading: which language a text is in (by its script) and whether a reply thinks and answers in the language
of its prompt. Standard library only: the kernel imports it too (as /k/exp9_lang.py), so the model has the exact
measure."""
import re
from collections import Counter

SCRIPTS = {"latin": [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F)], "kana": [(0x3040, 0x30FF)],
           "han": [(0x4E00, 0x9FFF)]}


def shares(text):
    """Share of each script among the text's letters (digits, symbols and spaces are not counted)."""
    c = Counter()
    for ch in text:
        o = ord(ch)
        for name, ranges in SCRIPTS.items():
            if any(a <= o <= b for a, b in ranges):
                c[name] += 1
                break
    total = sum(c.values())
    return {k: v / total for k, v in c.items()} if total else {}


def language(text):
    """en, ja, zh, mixed or none (no letters). Kana and Han together at least half of the letters: ja if kana is at
    least a tenth of the letters (Japanese prose is about half kana, Chinese has none), else zh. Latin at least half:
    en."""
    s = shares(text)
    if not s:
        return "none"
    kana, han = s.get("kana", 0), s.get("han", 0)
    if kana + han >= 0.5:
        return "ja" if kana >= 0.1 else "zh"
    return "en" if s.get("latin", 0) >= 0.5 else "mixed"


def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    try:
        return float(nums[-1].replace(",", "").rstrip(".")) if nums else None
    except ValueError:
        return None


def grade(text, lang=None, answer=None):
    """A reply (the text after the prompt's "<think>\\n"): its thinking is what precedes </think>, its answer what
    follows. in_lang: thinking and answer are both in `lang`; correct: the answer's last number is `answer`."""
    think, closed, final = text.partition("</think>")
    g = {"closed_think": bool(closed), "think_lang": language(think),
         "answer_lang": language(final) if closed else "none", "think_in_lang": None, "answer_in_lang": None,
         "pred": last_number(final) if closed else None, "correct": None}
    if lang:
        g["think_in_lang"] = g["think_lang"] == lang
        g["answer_in_lang"] = g["answer_lang"] == lang
    if answer is not None:
        g["correct"] = g["pred"] is not None and abs(g["pred"] - float(answer)) < 1e-6
    return g

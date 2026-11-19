"""exp10's script measure: is a reply written in Hindi (Devanagari)? Standard library only: the kernel imports it too
(as /k/exp10_lang.py), so the model has the exact measure."""
from collections import Counter

SCRIPTS = {"latin": [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F)], "devanagari": [(0x900, 0x97F)],
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
        else:
            if ch.isalpha():
                c["other"] += 1
    total = sum(c.values())
    return {k: v / total for k, v in c.items()} if total else {}


def in_hindi(text):
    """True if at least half of the text's letters are Devanagari; None if it has no letters."""
    s = shares(text)
    return (s.get("devanagari", 0) >= 0.5) if s else None


def first_letter(text):
    """The first of A, B, C, D in a reply to a multiple-choice question, or None."""
    for ch in text.strip()[:40]:
        if ch in "ABCD":
            return ch
    return None

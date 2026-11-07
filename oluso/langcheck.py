"""Lightweight orthographic check for Yoruba output.

Not a language identifier. It flags text that carries Yoruba orthography (dot-below
letters, tone marks) and common Yoruba function words, which is enough to tell
Yoruba output apart from English or Pidgin output in smoke tests.
"""

from __future__ import annotations

import re
import unicodedata

DOT_BELOW = "̣"
TONE_MARKS = {"̀", "́"}  # grave (low), acute (high)

# Common Yoruba function words, compared after stripping diacritics.
FUNCTION_WORDS = {
    "ni", "ti", "ati", "si", "ko", "je", "awon", "pe", "fun", "mo", "wa", "yii",
    "naa", "se", "lati", "kan", "won", "o", "fi", "lo", "tabi", "bi", "ninu",
    "nipa", "ile", "eko", "omo", "ilu", "oun", "re", "yin", "gbogbo", "sugbon",
}  # fmt: skip

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _strip_marks(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def yoruba_signals(text: str) -> dict:
    nfd = unicodedata.normalize("NFD", text)
    letters = sum(1 for c in nfd if c.isalpha())
    dot_below = nfd.count(DOT_BELOW)
    tones = sum(1 for c in nfd if c in TONE_MARKS)
    words = [w.lower() for w in _WORD.findall(_strip_marks(text))]
    fw = sum(1 for w in words if w in FUNCTION_WORDS)
    return {
        "letters": letters,
        "words": len(words),
        "dot_below": dot_below,
        "tone_marks": tones,
        "mark_density": (dot_below + tones) / letters if letters else 0.0,
        "function_word_ratio": fw / len(words) if words else 0.0,
    }


def looks_yoruba(text: str) -> bool:
    s = yoruba_signals(text)
    if s["words"] < 3:
        return False
    return s["mark_density"] >= 0.03 and s["function_word_ratio"] >= 0.12

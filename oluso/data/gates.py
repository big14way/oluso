"""Quality gates for machine-translated and rewritten prompts.

No single gate is trusted on its own. A row passes only if every applicable gate
passes; failures are kept with reasons so they can be reviewed or re-routed.
"""

from __future__ import annotations

import re

from sacrebleu.metrics import CHRF

_CHRF = CHRF()

# Signs that a generator refused or editorialised instead of translating/rewriting.
REFUSAL = re.compile(
    r"\b(i\s*(?:can(?:no|')t|won't|am (?:not able|unable))|sorry|as an ai|i must decline"
    r"|cannot (?:help|assist|provide)|not (?:able|allowed) to)\b",
    re.IGNORECASE,
)


# Nigerian Pidgin function words and frequent lexemes (BBC Pidgin and informal spellings).
PIDGIN_MARKERS = {
    "dey", "wey", "di", "na", "don", "dem", "una", "wetin", "abeg", "pikin", "pesin",
    "pipo", "sabi", "oda", "wit", "dia", "wen", "moni", "kain", "sef", "abi", "comot",
    "chop", "make", "fit", "am", "im", "e", "go", "say", "wahala", "tok", "waka", "small",
}  # fmt: skip
_WORD = re.compile(r"[a-z']+")


def pidgin_markers(text: str) -> tuple[int, float]:
    """Count Pidgin marker tokens and their share of all tokens.

    Several markers (make, fit, go, say, small, am) are also English words, so the
    density is only meaningful relative to the English source, never on its own.
    """
    words = _WORD.findall(text.lower())
    n = sum(w in PIDGIN_MARKERS for w in words)
    return n, n / len(words) if words else 0.0


def chrf(hyp: str, ref: str) -> float:
    return _CHRF.sentence_score(hyp, [ref]).score


def length_ratio(src: str, tgt: str) -> float:
    return len(tgt) / max(len(src), 1)


def check_translation(
    src: str,
    tgt: str,
    back: str | None,
    lid: list[tuple[str, float]],
    want_label: str | set[str],
    min_lid_prob: float = 0.5,
    ratio: tuple[float, float] = (0.5, 2.5),
    min_bt_chrf: float = 30.0,
) -> tuple[bool, list[str], dict]:
    reasons: list[str] = []
    wanted = {want_label} if isinstance(want_label, str) else want_label
    top_label, top_prob = lid[0] if lid else ("", 0.0)
    lr = length_ratio(src, tgt)
    bt = chrf(back, src) if back is not None else None
    if not tgt.strip():
        reasons.append("empty")
    if top_label not in wanted or top_prob < min_lid_prob:
        reasons.append(f"lid:{top_label}:{top_prob:.2f}")
    if not ratio[0] <= lr <= ratio[1]:
        reasons.append(f"length_ratio:{lr:.2f}")
    if REFUSAL.search(tgt) and not REFUSAL.search(src):
        reasons.append("refusal_text")
    if bt is not None and bt < min_bt_chrf:
        reasons.append(f"bt_chrf:{bt:.1f}")
    signals = {
        "pidgin_markers": pidgin_markers(tgt)[0] - pidgin_markers(src)[0],
        "lid_top": top_label,
        "lid_prob": round(top_prob, 3),
        "len_ratio": round(lr, 2),
        "bt_chrf": None if bt is None else round(bt, 1),
    }
    return (not reasons), reasons, signals

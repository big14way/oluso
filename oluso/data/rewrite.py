"""Few-shot rewriting into Nigerian Pidgin and Pidgin-English code-mix.

Runs a causal LM in raw completion mode (no chat template, so no assistant persona
and no safety preamble): the model continues a pattern of gold examples rather than
answering a request. Tiny Aya Global is used, not the evaluated Tiny Aya Earth, so
the model under test never writes its own test prompts.
"""

from __future__ import annotations

import json
from pathlib import Path

from oluso import config
from oluso.data.gates import REFUSAL, check_translation, chrf, length_ratio, pidgin_markers
from oluso.env import hf_token, pick_device, pick_dtype

GLOBAL = "CohereLabs/tiny-aya-global"
GLOBAL_REV = "af89d219b53ed9b13b8a4645f9c8028973510324"
GOLD = config.ROOT / "data" / "gold" / "pidgin_triples.jsonl"
FIELDS = {"pcm": ("pcm", "Nigerian Pidgin"), "pcm-en": ("pcm_en", "Pidgin-English mix")}


def load_gold(path: Path = GOLD) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fewshot_prompt(gold: list[dict], target: str, en: str) -> str:
    key, name = FIELDS[target]
    shots = [f"English: {g['en']}\n{name}: {g[key]}" for g in gold]
    return "\n\n".join([*shots, f"English: {en}\n{name}:"])


class Completer:
    def __init__(self, model_id: str, revision: str | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or pick_device()
        token = hf_token()
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, token=token, dtype=pick_dtype(self.device)
        ).to(self.device)
        self.model.eval()
        self._torch = torch

    def __call__(self, prompts: list[str], max_new_tokens: int = 80, batch_size: int = 8):
        out: list[str] = []
        for i in range(0, len(prompts), batch_size):
            enc = self.tok(prompts[i : i + batch_size], return_tensors="pt", padding=True).to(
                self.device
            )
            with self._torch.inference_mode():
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tok.pad_token_id,
                )
            texts = self.tok.batch_decode(
                gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            out.extend(t.strip().split("\n")[0].strip() for t in texts)
        return out


class MockCompleter:
    def __call__(self, prompts: list[str], **_) -> list[str]:
        # Echo the English line with a Pidgin frame so offline runs exercise the gates.
        return [
            "Abeg " + p.rsplit("English: ", 1)[1].split("\n")[0] + " wey dey work" for p in prompts
        ]


def load_completer(backend: str, device: str | None = None):
    if backend == "mock":
        return MockCompleter()
    if backend == "tiny":
        return Completer(config.TINY_TEST_MODEL, device=device or "cpu")
    if backend == "global":
        return Completer(GLOBAL, revision=GLOBAL_REV, device=device)
    raise ValueError(f"unknown backend {backend!r}")


def check_codemix(en: str, mix: str, pcm: str | None, lid) -> tuple[bool, list[str], dict]:
    """Gates for Pidgin-English code-mixed text: keeps English content, adds Pidgin."""
    reasons: list[str] = []
    top, prob = lid[0] if lid else ("", 0.0)
    to_en = chrf(mix, en)
    to_pcm = chrf(mix, pcm) if pcm else None
    markers = pidgin_markers(mix)[0] - pidgin_markers(en)[0]
    lr = length_ratio(en, mix)
    if not mix.strip():
        reasons.append("empty")
    if top not in {"eng_Latn", "pcm_Latn", "wes_Latn"}:
        reasons.append(f"lid:{top}:{prob:.2f}")
    if REFUSAL.search(mix) and not REFUSAL.search(en):
        reasons.append("refusal_text")
    if not 0.6 <= lr <= 2.0:
        reasons.append(f"length_ratio:{lr:.2f}")
    if markers < 1:
        reasons.append("no_pidgin_markers")
    if to_en >= 92:
        reasons.append(f"same_as_english:{to_en:.0f}")
    if to_en < 30:
        reasons.append(f"lost_english_content:{to_en:.0f}")
    if to_pcm is not None and to_pcm >= 95:
        reasons.append("same_as_pidgin")
    signals = {
        "lid_top": top,
        "lid_prob": round(prob, 3),
        "len_ratio": round(lr, 2),
        "chrf_to_en": round(to_en, 1),
        "chrf_to_pcm": None if to_pcm is None else round(to_pcm, 1),
        "pidgin_markers": markers,
    }
    return (not reasons), reasons, signals


def pick_pcm(en: str, candidates: list[tuple[str, str, str]], lid, accept: set[str]) -> dict:
    """Choose between Pidgin candidates [(origin, text, back_translation)].

    Passing candidates beat failing ones; ties go to more Pidgin markers, then to
    higher back-translation chrF (meaning preserved).
    """
    scored = []
    for origin, text, back in candidates:
        ok, reasons, sig = check_translation(en, text, back, lid.predict(text), accept)
        scored.append(
            (ok, sig["pidgin_markers"], sig["bt_chrf"] or 0.0, origin, text, back, reasons, sig)
        )
    scored.sort(key=lambda s: (s[0], s[1], s[2]), reverse=True)
    ok, _, _, origin, text, back, reasons, sig = scored[0]
    return {
        "prompt": text,
        "origin": origin,
        "back_en": back,
        "gate_pass": ok,
        "gate_reasons": reasons,
        **sig,
        "alternatives": [{"origin": s[3], "text": s[4], "gate_pass": s[0]} for s in scored[1:]],
    }

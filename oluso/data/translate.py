"""Machine translation backends (encoder-decoder MT models, which do not refuse).

- NLLB-200 for English to/from Hausa, Igbo and Yoruba.
- MAFAND M2M-100 models for English to/from Nigerian Pidgin. M2M-100 has no Pidgin
  code, so these checkpoints were fine-tuned with `__fr__` standing in for Pidgin.
"""

from __future__ import annotations

from dataclasses import dataclass

from oluso.env import pick_device

NLLB = "facebook/nllb-200-distilled-1.3B"
NLLB_CODES = {"en": "eng_Latn", "ha": "hau_Latn", "ig": "ibo_Latn", "yo": "yor_Latn"}

PCM_FWD = "masakhane/m2m100_418M_en_pcm_rel_news_ft"
PCM_BWD = "masakhane/m2m100_418M_pcm_en_rel_news_ft"
M2M_PCM_PROXY = "fr"


@dataclass(frozen=True)
class Direction:
    model_id: str
    src: str  # tokenizer source language code
    tgt_token: str  # token forced as the first decoder token


def direction(src: str, tgt: str, nllb_model: str = NLLB) -> Direction:
    if src == "en" and tgt == "pcm":
        return Direction(PCM_FWD, "en", f"__{M2M_PCM_PROXY}__")
    if src == "pcm" and tgt == "en":
        return Direction(PCM_BWD, M2M_PCM_PROXY, "__en__")
    if src in NLLB_CODES and tgt in NLLB_CODES:
        return Direction(nllb_model, NLLB_CODES[src], NLLB_CODES[tgt])
    raise ValueError(f"no MT route for {src}->{tgt}")


class Translator:
    def __init__(self, d: Direction, device: str | None = None, num_beams: int = 4):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.d = d
        self.device = device or pick_device()
        self.num_beams = num_beams
        self.tok = AutoTokenizer.from_pretrained(d.model_id, src_lang=d.src)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForSeq2SeqLM.from_pretrained(d.model_id, dtype=dtype).to(self.device)
        self.model.eval()
        self.tgt_id = self.tok.convert_tokens_to_ids(d.tgt_token)
        if self.tgt_id == self.tok.unk_token_id:
            raise ValueError(f"{d.tgt_token} is not a token of {d.model_id}")
        self._torch = torch

    def __call__(self, texts: list[str], batch_size: int = 16, max_new_tokens: int = 256):
        out: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True).to(
                self.device
            )
            with self._torch.inference_mode():
                gen = self.model.generate(
                    **enc,
                    forced_bos_token_id=self.tgt_id,
                    num_beams=self.num_beams,
                    max_new_tokens=max_new_tokens,
                )
            out.extend(s.strip() for s in self.tok.batch_decode(gen, skip_special_tokens=True))
        return out


class MockTranslator:
    """Offline stand-in: tags the text with the target language."""

    def __init__(self, d: Direction):
        self.d = d

    def __call__(self, texts: list[str], **_) -> list[str]:
        return [f"[{self.d.tgt_token}] {t}" for t in texts]


def load(src: str, tgt: str, mock: bool = False, **kw):
    d = direction(src, tgt)
    return MockTranslator(d) if mock else Translator(d, **kw)


# Back-translation uses a different model family from the forward pass, so that
# a systematic forward error is not undone by the same model in reverse.
M2M_BACK = "facebook/m2m100_418M"


def back_direction(lang: str) -> Direction:
    if lang == "pcm":
        return direction("pcm", "en")
    if lang in ("ha", "ig", "yo"):
        return Direction(M2M_BACK, lang, "__en__")
    raise ValueError(f"no back-translation route for {lang}")


def load_back(lang: str, mock: bool = False, **kw):
    d = back_direction(lang)
    return MockTranslator(d) if mock else Translator(d, **kw)

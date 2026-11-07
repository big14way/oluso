"""Language identification with GlotLID (covers eng, yor, ibo, hau and pcm)."""

from __future__ import annotations

GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_FILE = "model_v4.bin"

# Map our settings to GlotLID labels.
LID_LABEL = {
    "en": "eng_Latn",
    "yo": "yor_Latn",
    "ig": "ibo_Latn",
    "ha": "hau_Latn",
    "pcm": "pcm_Latn",
}


# Labels accepted as a match. GlotLID often labels BBC-style Nigerian Pidgin as
# Cameroonian Pidgin (wes_Latn), a closely related English-lexifier creole.
ACCEPT = {k: {v} for k, v in LID_LABEL.items()} | {"pcm": {"pcm_Latn", "wes_Latn"}}


class GlotLID:
    def __init__(self):
        import fasttext
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(GLOTLID_REPO, GLOTLID_FILE)
        self.model = fasttext.load_model(path)

    def predict(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        # Call the C++ binding directly: fasttext 0.9.3's Python wrapper breaks on numpy 2.
        pairs = self.model.f.predict(" ".join(text.split()), k, 0.0, "strict")
        return [(label.removeprefix("__label__"), float(prob)) for prob, label in pairs]


class MockLID:
    def __init__(self, label: str = "pcm_Latn"):
        self.label = label

    def predict(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        return [(self.label, 1.0)][:k]

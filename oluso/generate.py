"""Chat generation backends.

Backends:
  hf    the target model (Tiny Aya Earth), needs licence acceptance and HF_TOKEN
  tiny  ungated random-weight cohere2 model; same code path, meaningless text
  mock  no download, deterministic canned replies; for dry runs without credentials
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from oluso import config
from oluso.env import hf_token, pick_device, pick_dtype


class ModelAccessError(RuntimeError):
    pass


@dataclass
class Generation:
    prompt: str
    output: str
    new_tokens: int
    seconds: float


@dataclass
class MockGenerator:
    name: str = "mock"
    model_id: str = "mock"
    revision: str = "none"
    device: str = "none"
    replies: dict[str, str] = field(
        default_factory=lambda: {
            "yo": "Èyí jẹ́ ìdáhùn àpẹẹrẹ ní èdè Yorùbá, kì í ṣe ìdáhùn gidi láti inú àwòṣe.",
            "en": "This is a canned mock reply, not real model output.",
        }
    )

    def chat(self, prompts: list[str], max_new_tokens: int = 128, lang: str = "en") -> list:
        reply = self.replies.get(lang, self.replies["en"])
        return [Generation(p, reply, len(reply.split()), 0.0) for p in prompts]


class HFGenerator:
    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str | None = None,
        name: str = "hf",
    ):
        import torch
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.model_id = model_id
        self.revision = revision or "main"
        self.device = device or pick_device()
        dtype = pick_dtype(self.device)
        token = hf_token()
        try:
            self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision, token=token)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, token=token, dtype=dtype
            ).to(self.device)
        except (GatedRepoError, RepositoryNotFoundError, OSError) as exc:
            if isinstance(exc, OSError) and "gated" not in str(exc).lower():
                raise
            raise ModelAccessError(
                f"Cannot download {model_id}. Accept the licence at "
                f"https://huggingface.co/{model_id} with the account whose token you use, "
                "then set HF_TOKEN in .env or run `hf auth login`. "
                f"Token currently {'present' if token else 'missing'}."
            ) from exc
        self.model.eval()
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self._torch = torch

    def _encode(self, prompts: list[str]):
        texts = [
            self.tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in prompts
        ]
        return self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(
            self.device
        )

    def chat(self, prompts: list[str], max_new_tokens: int = 128, lang: str = "en") -> list:
        batch = self._encode(prompts)
        start = time.perf_counter()
        with self._torch.inference_mode():
            out = self.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tok.pad_token_id,
            )
        seconds = time.perf_counter() - start
        new = out[:, batch["input_ids"].shape[1] :]
        results = []
        for prompt, ids in zip(prompts, new, strict=True):
            n = int((ids != self.tok.pad_token_id).sum())
            text = self.tok.decode(ids, skip_special_tokens=True).strip()
            results.append(Generation(prompt, text, n, seconds / len(prompts)))
        return results


def load_generator(backend: str, device: str | None = None):
    if backend == "mock":
        return MockGenerator()
    if backend == "tiny":
        return HFGenerator(config.TINY_TEST_MODEL, device=device or "cpu", name="tiny")
    if backend == "hf":
        return HFGenerator(config.MODEL_ID, revision=config.MODEL_REVISION, device=device)
    raise ValueError(f"unknown backend {backend!r}; expected hf, tiny or mock")

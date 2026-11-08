"""A drop-in guard for Tiny Aya Earth: the selected CAST steering and the calibrating temperature.

    from oluso.guard import Guard
    g = Guard.load("dist/guard")          # or a Hugging Face repo id
    g.chat(["Abeg, how I fit ..."])       # steered answers (the gate decides per prompt)
    g.multiple_choice(question, options)  # calibrated option probabilities

A guard directory holds `guard.json` (model revision, layer, alpha, temperature, gate
settings, provenance) and `steering.pt` (the one behaviour vector and the gate's condition
vector). Nothing else is needed: no fine-tuned weights, the base model is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oluso import config, steer
from oluso.steer import Arm

FORMAT = "oluso-guard/1"


def export(run_dir: Path, out_dir: Path, temperature: float, provenance: dict | None = None):
    """Write a guard from a P3 run (artifacts.pt + selection.json) and a fitted temperature."""
    import torch

    art = steer.load_artifacts(run_dir / "artifacts.pt")
    sel = json.loads((run_dir / "selection.json").read_text())["selected"]
    c = art["condition"]
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"vector": art["vectors"][sel["layer"]], "condition_vector": c.condition_vector},
               out_dir / "steering.pt")  # fmt: skip
    meta = {
        "format": FORMAT,
        "model": config.MODEL_ID,
        "revision": config.MODEL_REVISION,
        "layer": sel["layer"],
        "alpha": sel["alpha"],
        "gate": {"layer": c.condition_layer, "threshold": c.threshold,
                 "comparator": c.comparator, "pool": c.pool, "heldout_f1": c.f1},
        "temperature": temperature,
        "provenance": provenance or {},
    }  # fmt: skip
    (out_dir / "guard.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


@dataclass
class Guard:
    gen: object
    meta: dict
    art: dict

    @classmethod
    def load(cls, path: str | Path, backend: str = "hf", device: str | None = None) -> Guard:
        import torch
        from safetune.steer import CASTCondition

        from oluso.generate import load_generator

        d = Path(path)
        if not d.exists():  # a Hugging Face repo id
            from huggingface_hub import snapshot_download

            d = Path(snapshot_download(str(path)))
        meta = json.loads((d / "guard.json").read_text())
        if meta.get("format") != FORMAT:
            raise ValueError(f"not an oluso guard: {d}")
        raw = torch.load(d / "steering.pt", weights_only=True)
        g = meta["gate"]
        cond = CASTCondition(condition_vector=raw["condition_vector"], condition_layer=g["layer"],
                             threshold=g["threshold"], comparator=g["comparator"],
                             f1=g.get("heldout_f1", 0.0), pool=g["pool"])  # fmt: skip
        art = {"vectors": {meta["layer"]: raw["vector"]}, "condition": cond}
        return cls(load_generator(backend, device=device), meta, art)

    @property
    def arm(self) -> Arm:
        return Arm("cast", self.meta["layer"], self.meta["alpha"])

    def chat(self, prompts: list[str], max_new_tokens: int = 256) -> list[str]:
        with steer.steered(self.gen, self.arm, self.art):
            return [g.output for g in self.gen.chat(prompts, max_new_tokens)]

    def multiple_choice(self, question: str, options: list[str]) -> dict:
        """Option probabilities from the answer-letter distribution, temperature-scaled."""
        from oluso.calibration import LETTERS, letter_scores, softmax

        with steer.steered(self.gen, self.arm, self.art):
            s, mass = letter_scores(self.gen.model, self.gen.tok, question, options)
        p = softmax(np.asarray(s), self.meta["temperature"])
        best = int(p.argmax())
        probs = {LETTERS[i]: float(x) for i, x in enumerate(p)}
        return {"answer": LETTERS[best], "confidence": float(p[best]), "probs": probs,
                "letter_mass": mass}  # fmt: skip

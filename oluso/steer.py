"""P3 repair: conditional activation steering (CAST) with SafeTune, and its controls.

Behaviour vector (CAA, `safetune.steer.extract_caa_vectors`): the difference of means of
the last prompt token's residual stream, harmful minus harmless, over the steering fit
split in all five languages, at one layer. Adding alpha times it moves the model towards
the state it is in when it has recognised a harmful request; alpha = 1 adds one full
mean difference.

Gate (CAST, Wu et al. 2024, `safetune.steer.fit_cast_condition`): a condition vector
fitted on the same prompts, with layer, threshold and comparator chosen by held-out F1.
The behaviour vector is added only to prompts whose gate fires, for the whole generation.

Arms, at the same layer and alpha:
- cast:   the gate and the real vector (the repair).
- random: the same gate with a random direction of equal norm (seeded): is it the
          direction that helps, or any push at that layer?
- add:    the real vector on every prompt, no gate (plain CAA): what the gate buys.

Selection uses the dev split only; the locked test set is run once per arm.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from oluso import config
from oluso.data import testset

FIT_PATH = config.ROOT / "data" / "fit" / "fit.jsonl"
KINDS = ("cast", "random", "add")


@dataclass(frozen=True)
class Arm:
    kind: str  # cast | random | add | base
    layer: int = -1
    alpha: float = 0.0
    seed: int = 0

    @property
    def name(self) -> str:
        return "base" if self.kind == "base" else f"{self.kind}-L{self.layer}-a{self.alpha:g}"

    def to_dict(self) -> dict:
        return asdict(self) | {"name": self.name}


BASE = Arm("base")


def fit_prompts(split: str = "fit") -> tuple[list[str], list[str]]:
    """(harmful, harmless) prompt texts of one split of the steering fit set, all languages."""
    rows = [r for r in testset.read_jsonl(FIT_PATH) if r["split"] == split]
    harm = [r["text"] for r in rows if r["side"] == "harm"]
    safe = [r["text"] for r in rows if r["side"] == "harmless"]
    return harm, safe


def fit_artifacts(model, tok, layers: list[int], harm: list[str], safe: list[str],
                  gate_layers: list[int] | None = None, log=print) -> dict:  # fmt: skip
    """CAA vectors at each candidate layer, and one CAST gate."""
    from safetune.steer import CAAConfig, extract_caa_vectors, fit_cast_condition

    vectors = {}
    for layer in layers:  # one layer per call: the extractor keeps full sequences in RAM
        cfg = CAAConfig(target_layers=[layer], pool_method="last_token")
        vectors[layer] = extract_caa_vectors(model, tok, harm, safe, cfg)[layer].float().cpu()
        if log:
            log(f"CAA vector at layer {layer}: norm {vectors[layer].norm():.2f}")
    cond = fit_cast_condition(model, harm, safe, tok, candidate_layers=gate_layers, pool="last")
    if log:
        log(f"CAST gate: layer {cond.condition_layer}, threshold {cond.threshold:.2f}, "
            f"{cond.comparator}, held-out F1 {cond.f1:.3f}")  # fmt: skip
    return {"vectors": vectors, "condition": cond}


def save_artifacts(art: dict, path: Path) -> None:
    import torch

    c = art["condition"]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": art["vectors"],
                "condition": {"condition_vector": c.condition_vector, "condition_layer":
                              c.condition_layer, "threshold": c.threshold,
                              "comparator": c.comparator, "f1": c.f1, "pool": c.pool}},
               path)  # fmt: skip


def load_artifacts(path: Path) -> dict:
    import torch
    from safetune.steer import CASTCondition

    raw = torch.load(path, weights_only=True)
    return {"vectors": raw["vectors"], "condition": CASTCondition(**raw["condition"])}


def random_like(v, seed: int):
    """A random direction with the same norm as v (the matched control)."""
    import torch

    g = torch.Generator().manual_seed(seed)
    r = torch.randn(v.shape, generator=g, dtype=torch.float32)
    return r / r.norm() * v.float().norm()


class GatedCall:
    """A CASTModel whose plain call also goes through the gate.

    CASTModel.generate gates, but CASTModel.__call__ runs the bare model; option and
    letter scoring call the model directly, so they need this to see the steered model.
    """

    def __init__(self, cast):
        self.__dict__["cast"] = cast

    def __call__(self, *args, **kwargs):
        return self.cast.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.cast.generate(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__dict__["cast"], name)


def wrap(model, arm: Arm, art: dict):
    from safetune.steer import CAAModel, CASTModel

    if arm.kind == "base":
        return model
    v = art["vectors"][arm.layer]
    if arm.kind == "random":
        v = random_like(v, arm.seed)
    if arm.kind == "add":
        return CAAModel(model, {arm.layer: v}, strength=arm.alpha)
    if arm.kind not in KINDS:
        raise ValueError(f"unknown arm kind {arm.kind!r}")
    return GatedCall(CASTModel(model, {arm.layer: v}, condition=art["condition"],
                               alpha=arm.alpha, per_prompt_gate=True))  # fmt: skip


@contextmanager
def steered(gen, arm: Arm, art: dict | None):
    """Swap the generator's model for the steered wrapper inside the block."""
    if arm.kind == "base" or not hasattr(gen, "model"):
        yield gen
        return
    base = gen.model
    gen.model = wrap(base, arm, art)
    try:
        yield gen
    finally:
        gen.model = base


def gate_report(gen, art: dict, rows: list[dict], batch_size: int = 16) -> dict:
    """How often the gate fires on each (setting, label) group of `rows`."""
    from collections import defaultdict

    from safetune.steer import CASTModel

    cast = CASTModel(gen.model, {}, condition=art["condition"], per_prompt_gate=True)
    hits: dict = defaultdict(lambda: [0, 0])
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        enc = gen._encode([r["prompt"] for r in chunk])
        fires, _ = cast._gate_fires(enc["input_ids"], attention_mask=enc["attention_mask"])
        for r, f in zip(chunk, fires, strict=True):
            h = hits[f"{r['setting']}/{r['label']}"]
            h[0] += int(f)
            h[1] += 1
    return {k: {"fired": a, "n": n, "rate": round(a / n, 4)} for k, (a, n) in sorted(hits.items())}

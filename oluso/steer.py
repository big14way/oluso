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


class LastLogits:
    """The model, except that a plain call computes logits for the last position only.

    SafeTune's vector extraction and CAST's gate pass call the model just to read hidden
    states through hooks; with a 262k vocabulary, logits at every position of a batch do
    not fit in GPU memory (16 prompts x 390 tokens = 6 GiB in float32). An explicit
    `logits_to_keep` (option and letter scoring) is respected; `generate` is untouched.
    """

    def __init__(self, model):
        self.__dict__["_m"] = model

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("logits_to_keep", 1)
        return self.__dict__["_m"](*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.__dict__["_m"], name)


def fit_prompts(split: str = "fit") -> tuple[list[str], list[str]]:
    """(harmful, harmless) prompt texts of one split of the steering fit set, all languages."""
    rows = [r for r in testset.read_jsonl(FIT_PATH) if r["split"] == split]
    harm = [r["text"] for r in rows if r["side"] == "harm"]
    safe = [r["text"] for r in rows if r["side"] == "harmless"]
    return harm, safe


@contextmanager
def last_position_capture():
    """SafeTune's extractor hook stores every position of every batch in host RAM and pools
    afterwards; with last-token pooling only the last position is ever used. Inside this
    block the hook stores just that position: the same vectors, a few MB instead of ~2 GB
    per layer and side (which ran a 15 GB job machine out of memory)."""
    from safetune.core.runtime.inference.vector_extraction import SteeringVectorExtractor

    original = SteeringVectorExtractor._make_hook

    def _make_hook(self, layer_idx: int):
        if self.config.pool_method != "last_token":
            return original(self, layer_idx)

        def hook_fn(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            self._activations.setdefault(layer_idx, []).append(h[:, -1:, :].detach().cpu())

        return hook_fn

    SteeringVectorExtractor._make_hook = _make_hook
    try:
        yield
    finally:
        SteeringVectorExtractor._make_hook = original


@contextmanager
def compact_gate_pooling():
    """CAST's gate fit copies each prompt's whole (seq, hidden) activation to host RAM at
    every candidate layer, then pools it with `h[-1, :]`, a view that keeps the whole buffer
    alive: ~3 MB per layer per prompt, 19 layers x 1,292 prompts ~ 80 GB. Even with the view
    copied, ~25 variable-size multi-MB host buffers per prompt fragment the heap (+1 GB per
    15 s on the job machine). Inside this block the gate's collector pools on the device and
    moves only the pooled vector: same prompts, encoding and pooling, one prompt at a time,
    so the gate fit is unchanged."""
    import torch
    from safetune._refusal_helpers import _encode_prompts, _get_decoder_layers
    from safetune.interventions.steer import cast

    original = cast._collect_pooled_hidden

    def _collect_pooled_hidden(model, tokenizer, prompts, layer_indices, pool="mean",
                               batch_size=8, chat_template=True):  # fmt: skip
        layers = _get_decoder_layers(model)
        for li in layer_indices:
            if not 0 <= li < len(layers):
                raise IndexError(f"condition layer {li} out of range ({len(layers)} layers)")
        captured: dict = {}

        def mk(li):
            def hook(_m, _i, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[li] = cast._pool_hidden(h[0].detach().float(), pool).cpu()

            return hook

        handles = [layers[li].register_forward_hook(mk(li)) for li in layer_indices]
        out: dict = {li: [] for li in layer_indices}
        try:
            model.eval()
            device = next(model.parameters()).device
            for p in prompts:
                captured.clear()
                enc = _encode_prompts(tokenizer, [p], chat_template, return_tensors="pt",
                                      padding=True, truncation=True)  # fmt: skip
                enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in dict(enc).items()}
                with torch.no_grad():
                    model(**enc)
                for li in layer_indices:
                    if li in captured:
                        out[li].append(captured[li])
        finally:
            for hd in handles:
                hd.remove()
        if any(not v for v in out.values()):
            raise RuntimeError("gate fit: no activations captured")
        return {li: torch.stack(v, dim=0) for li, v in out.items()}

    cast._collect_pooled_hidden = _collect_pooled_hidden
    try:
        yield
    finally:
        cast._collect_pooled_hidden = original


def fit_artifacts(model, tok, layers: list[int], harm: list[str], safe: list[str],
                  gate_layers: list[int] | None = None, log=print) -> dict:  # fmt: skip
    """CAA vectors at each candidate layer, and one CAST gate."""
    from safetune.steer import CAAConfig, extract_caa_vectors, fit_cast_condition

    model = LastLogits(model)
    cfg = CAAConfig(target_layers=list(layers), pool_method="last_token")
    with last_position_capture():  # one pass for all layers
        raw = extract_caa_vectors(model, tok, harm, safe, cfg)
    vectors = {layer: raw[layer].float().cpu() for layer in layers}
    if log:
        for layer in layers:
            log(f"CAA vector at layer {layer}: norm {vectors[layer].norm():.2f}")
    with compact_gate_pooling():
        cond = fit_cast_condition(model, harm, safe, tok, candidate_layers=gate_layers,
                                  pool="last")  # fmt: skip
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
    model = LastLogits(model)
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

    cast = CASTModel(LastLogits(gen.model), {}, condition=art["condition"], per_prompt_gate=True)
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

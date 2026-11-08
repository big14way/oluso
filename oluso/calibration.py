"""Multiple-choice confidence, calibration metrics and per-language temperature scaling.

Scoring, two views of the same item (chat template applied):
- letter: the probability of each option letter as the first token of the reply, after
  an instruction to answer with the letter only (the usual multiple-choice protocol;
  one forward pass, no length effects). letter_mass records how much probability the
  model put on the letters at all.
- option_sum: the summed log-probability of each option's text as the reply,
  renormalised over the options. Not length-normalised: dividing by bytes (tried first)
  made the model pick the longest option 55-58% of the time and turned confidence
  into an anti-signal (AUROC below 0.5), so it is not used.

Temperature: one scalar T per language, fitted by minimising NLL (not ECE) with a
bounded search over log T, and evaluated out of fold (k-fold cross-fitting) so
no item is scored with a temperature fitted on itself. A scalar T never changes
the argmax, so accuracy is unchanged by construction.

ECE with 15 equal-width bins is biased upward at small n, so it is always reported
next to its noise floor: the ECE a perfectly calibrated model with the same
confidences would get at the same n.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

LETTERS = "ABCDEFGHIJ"


# --------------------------------------------------------------------------- scoring


def mc_prompt(question: str, options: Sequence[str], subject: str | None = None) -> str:
    head = f"{subject}\n\n" if subject else ""
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"{head}{question}\n\n{opts}"


LETTER_INSTRUCTION = "Answer with the letter of the correct option only."


def letter_token_ids(tok, n: int) -> list[list[int]]:
    """Single-token spellings of each option letter ("A", " A"), for the first reply token."""
    ids = []
    for letter in LETTERS[:n]:
        cands = {tuple(tok(v, add_special_tokens=False).input_ids) for v in (letter, f" {letter}")}
        ids.append(sorted({c[0] for c in cands if len(c) == 1}))
    return ids


def letter_scores(model, tok, question: str, options: Sequence[str], subject: str | None = None):
    """Log-probability of each option letter as the first token of the assistant reply.

    Returns (scores, letter_mass): scores are log p(letter) (summed over its spellings),
    letter_mass is the total probability on the option letters before renormalising.
    """
    import torch

    prompt = f"{mc_prompt(question, options, subject)}\n\n{LETTER_INSTRUCTION}"
    text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True)  # fmt: skip
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(input_ids=ids.to(device), logits_to_keep=1).logits[0, -1]
    logp = torch.log_softmax(logits.float(), dim=-1)
    letters = letter_token_ids(tok, len(options))
    scores = np.array([torch.logsumexp(logp[t], 0).item() for t in letters])
    return scores, float(np.exp(scores).sum())


def letter_scores_batch(model, tok, items: Sequence[tuple[str, Sequence[str]]],
                        batch_size: int = 16) -> list[tuple[np.ndarray, float]]:  # fmt: skip
    """letter_scores for many (question, options) at once: left-padded, last position only."""
    import torch

    out = []
    device = next(model.parameters()).device
    prev = tok.padding_side
    tok.padding_side = "left"
    try:
        for i in range(0, len(items), batch_size):
            chunk = items[i : i + batch_size]
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": f"{mc_prompt(q, o)}\n\n{LETTER_INSTRUCTION}"}],
                tokenize=False, add_generation_prompt=True) for q, o in chunk]  # fmt: skip
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
            with torch.inference_mode():
                logits = model(input_ids=enc.input_ids.to(device),
                               attention_mask=enc.attention_mask.to(device),
                               logits_to_keep=1).logits[:, -1]  # fmt: skip
            logp = torch.log_softmax(logits.float(), dim=-1)
            for row, (_, opts) in zip(logp, chunk, strict=True):
                letters = letter_token_ids(tok, len(opts))
                s = np.array([torch.logsumexp(row[t], 0).item() for t in letters])
                out.append((s, float(np.exp(s).sum())))
    finally:
        tok.padding_side = prev
    return out


def option_scores(model, tok, question: str, options: Sequence[str], subject: str | None = None):
    """Summed log-probability of each option's text as the assistant reply."""
    import torch

    prefix = tok.apply_chat_template(
        [{"role": "user", "content": mc_prompt(question, options, subject)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_ids = tok(prefix, add_special_tokens=False, return_tensors="pt").input_ids[0]
    opt_ids = [tok(o, add_special_tokens=False).input_ids for o in options]
    seqs = [torch.cat([prefix_ids, torch.tensor(ids, dtype=prefix_ids.dtype)]) for ids in opt_ids]
    width = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    input_ids = torch.full((len(seqs), width), pad, dtype=prefix_ids.dtype)
    attn = torch.zeros_like(input_ids)
    for i, s in enumerate(seqs):  # right padding keeps the prefix positions aligned
        input_ids[i, : len(s)] = s
        attn[i, : len(s)] = 1
    device = next(model.parameters()).device
    p = len(prefix_ids)
    keep = width - (p - 1)  # only positions that predict option tokens (saves vocab x prefix)
    with torch.inference_mode():
        logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                       logits_to_keep=keep).logits  # fmt: skip
    logp = torch.log_softmax(logits.float(), dim=-1)
    offset = width - logits.shape[1]  # absolute position of logits[:, 0]
    scores = []
    for i, ids in enumerate(opt_ids):
        idx = torch.arange(p - 1 - offset, p - 1 - offset + len(ids), device=device)
        target = torch.tensor(ids, device=device)
        scores.append(logp[i, idx, target].sum().item())
    return np.array(scores)


# --------------------------------------------------------------------------- metrics


def softmax(scores: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = scores / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def ece_equal_mass(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """ECE with bins holding equal numbers of items (less sensitive to sparse bins)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(conf)
    total = 0.0
    for chunk in np.array_split(order, min(n_bins, len(conf))):
        if len(chunk):
            total += len(chunk) / len(conf) * abs(conf[chunk].mean() - correct[chunk].mean())
    return float(total)


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def noise_floor_ece(conf: np.ndarray, n_bins: int = 15, n_sim: int = 500, seed: int = 0) -> float:
    """Mean ECE of a perfectly calibrated model with these confidences, at this n."""
    rng = np.random.default_rng(seed)
    conf = np.asarray(conf, float)
    sims = [ece(conf, rng.random(len(conf)) < conf, n_bins) for _ in range(n_sim)]
    return float(np.mean(sims))


def summarize(scores: np.ndarray, y: np.ndarray, T: float = 1.0, n_bins: int = 15) -> dict:
    return summarize_probs(softmax(scores, T), y, n_bins)


# --------------------------------------------------------------------------- temperature


def fit_temperature(scores: np.ndarray, y: np.ndarray, bounds=(0.05, 50.0)) -> tuple[float, bool]:
    """NLL-optimal scalar temperature; second value flags a fit at the search bound."""
    from scipy.optimize import minimize_scalar

    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    res = minimize_scalar(
        lambda lt: nll(softmax(scores, math.exp(lt)), y), bounds=(lo, hi), method="bounded"
    )
    at_bound = min(abs(res.x - lo), abs(res.x - hi)) < 0.01 * (hi - lo)
    return float(math.exp(res.x)), bool(at_bound)


def crossfit_probs(scores: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0):
    """Out-of-fold calibrated probabilities and the per-fold temperatures."""
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(y)), k)
    probs = np.zeros_like(scores, dtype=float)
    temps = []
    for f in folds:
        train = np.setdiff1d(np.arange(len(y)), f)
        T, _ = fit_temperature(scores[train], y[train])
        temps.append(T)
        probs[f] = softmax(scores[f], T)
    return probs, temps


def summarize_probs(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict:
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    return {
        "n": int(len(y)),
        "accuracy": float(correct.mean()),
        "mean_conf": float(conf.mean()),
        "ece15": ece(conf, correct, n_bins),
        "ece15_equal_mass": ece_equal_mass(conf, correct, n_bins),
        "ece15_noise_floor": noise_floor_ece(conf, n_bins),
        "brier": brier(probs, y),
        "nll": nll(probs, y),
    }


# --------------------------------------------------------------------------- uncertainty


def bootstrap_ci(
    fn: Callable[[np.ndarray], float], n: int, n_boot: int = 1000, seed: int = 0, alpha=0.05
) -> tuple[float, float]:
    """Percentile CI for a statistic of item indices (fn receives resampled indices)."""
    rng = np.random.default_rng(seed)
    stats = [fn(rng.integers(0, n, n)) for _ in range(n_boot)]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))

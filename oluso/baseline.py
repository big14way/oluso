"""Baseline runs of the target model: answers on the locked test set and the steering dev
split, and AfriMMLU option scores for calibration.

Generation goes through AuditKIT (`ak.generate`) with a plain `list[str] -> list[str]`
callable, so base, steered and random-control runs share one code path and differ only in
the model behind the callable. Decoding is greedy with a fixed `max_new_tokens`, and the
model's default chat template (including its built-in safety preamble) is used unchanged
in every run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

from oluso import config
from oluso.data import testset

P2_DIR = config.RUNS_DIR / "p2"
MAX_NEW_TOKENS = 256

AFRIMMLU = "masakhane/afrimmlu"
AFRIMMLU_REV = "96f247619673906ae4c321a3232afc67b98ea57e"
AFRIMMLU_LANGS = {"en": "eng", "ha": "hau", "ig": "ibo", "yo": "yor"}

HARMFUL_LABELS = {"harm", "culture"}  # TukaBench `culture` items are localised harmful requests


def eval_rows(include_dev: bool = True) -> list[dict]:
    """Locked test rows (split "test") plus the steering fit set's dev rows (split "dev")."""
    rows = [
        {**r, "split": "test"}
        for part in ("tukabench", "pidgin")
        for r in testset.read_jsonl(testset.TEST_DIR / f"{part}.jsonl")
    ]
    if include_dev:
        for r in testset.read_jsonl(config.ROOT / "data" / "fit" / "fit.jsonl"):
            if r["split"] != "dev":
                continue
            rows.append({
                "id": r["id"], "setting": r["setting"],
                "label": "harm" if r["side"] == "harm" else "benign",
                "index": int(r["id"].rsplit(":", 1)[1]), "prompt": r["text"],
                "goal_en": r["en"], "source": r["source"], "split": "dev",
            })  # fmt: skip
    return rows


def make_chat_fn(gen, batch_size: int = 32, max_new_tokens: int = MAX_NEW_TOKENS, log=print):
    """Wrap a generator as AuditKIT's callable model.

    AuditKIT hands a callable the whole dataset at once, so batching happens here:
    longest prompts first (less padding), original order restored. New-token counts
    are kept on the function as `fn.new_tokens[prompt]`.
    """

    def fn(prompts: list[str]) -> list[str]:
        order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
        out: list[str] = [""] * len(prompts)
        start = time.perf_counter()
        for b, s in enumerate(range(0, len(order), batch_size)):
            idx = order[s : s + batch_size]
            for i, g in zip(idx, gen.chat([prompts[i] for i in idx], max_new_tokens), strict=True):
                out[i] = g.output
                fn.new_tokens[prompts[i]] = g.new_tokens
            if log and (b % 5 == 0 or s + batch_size >= len(order)):
                done = min(s + batch_size, len(order))
                log(f"  generated {done}/{len(order)} in {time.perf_counter() - start:.0f}s")
        return out

    fn.new_tokens = {}
    return fn


def template_fingerprint(gen) -> dict:
    """Hash of the rendered chat template around a placeholder prompt."""
    tok = getattr(gen, "tok", None)
    if tok is None:
        return {"sha256": None, "chars": 0}
    text = tok.apply_chat_template(
        [{"role": "user", "content": "<<PROMPT>>"}], tokenize=False, add_generation_prompt=True
    )
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "chars": len(text),
        "prefix_tokens": len(tok(text, add_special_tokens=False).input_ids),
    }


def batch_check(gen, prompts: list[str], max_new_tokens: int = 64) -> dict:
    """Do batched (left-padded) and one-at-a-time greedy outputs agree?"""
    batched = [g.output for g in gen.chat(prompts, max_new_tokens)]
    single = [gen.chat([p], max_new_tokens)[0].output for p in prompts]
    same = [a == b for a, b in zip(batched, single, strict=True)]
    prefix = [
        len(_common_prefix(a.split(), b.split())) / max(len(b.split()), 1)
        for a, b in zip(batched, single, strict=True)
    ]
    return {
        "n": len(prompts),
        "identical": sum(same),
        "mean_shared_prefix": sum(prefix) / len(prefix),
    }


def _common_prefix(a: list, b: list) -> list:
    out = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        out.append(x)
    return out


def generate(
    rows: list[dict],
    gen,
    batch_size: int = 32,
    max_new_tokens: int = MAX_NEW_TOKENS,
    log: Callable | None = print,
) -> tuple[list[dict], dict]:
    """Answer every row through AuditKIT; returns (rows with `response`, run metadata)."""
    import auditkit as ak

    samples = [
        ak.Sample(input=r["prompt"], id=r["id"], task=f"{r['setting']}/{r['label']}") for r in rows
    ]
    fn = make_chat_fn(gen, batch_size, max_new_tokens, log=log)
    start = time.perf_counter()
    answered = ak.generate(samples, fn)
    seconds = time.perf_counter() - start
    by_id = {s.id: s.actual_output or "" for s in answered}
    out = []
    for r in rows:
        n = fn.new_tokens.get(r["prompt"])
        out.append({**r, "response": by_id[r["id"]], "new_tokens": n,
                    "hit_max": n is not None and n >= max_new_tokens})  # fmt: skip
    meta = {
        "model": getattr(gen, "model_id", "mock"),
        "revision": getattr(gen, "revision", None),
        "device": getattr(gen, "device", None),
        "dtype": str(getattr(getattr(gen, "model", None), "dtype", None)),
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "decoding": "greedy",
        "template": template_fingerprint(gen),
        "rows": len(out),
        "seconds": round(seconds, 1),
    }
    return out, meta


# --------------------------------------------------------------------------- AfriMMLU


def afrimmlu_items(splits=("validation", "test"), langs=AFRIMMLU_LANGS) -> list[dict]:
    """Parallel AfriMMLU items; `k` is the item's position, shared across languages."""
    from datasets import load_dataset

    items = []
    for lang, cfg in langs.items():
        ds = load_dataset(AFRIMMLU, cfg, revision=AFRIMMLU_REV)
        for split in splits:
            for k, r in enumerate(ds[split]):
                items.append({
                    "id": f"afrimmlu:{lang}:{split}:{k:03d}", "lang": lang, "split": split,
                    "k": k, "subject": r["subject"], "question": r["question"],
                    "choices": ast.literal_eval(r["choices"]), "answer": "ABCD".index(r["answer"]),
                })  # fmt: skip
    return items


def score_afrimmlu(items: list[dict], gen, log: Callable | None = print) -> list[dict]:
    """Byte-normalised option log-probs under the chat template (see oluso.calibration)."""
    from oluso.calibration import option_scores

    out = []
    start = time.perf_counter()
    for i, it in enumerate(items):
        s = option_scores(gen.model, gen.tok, it["question"], it["choices"])
        out.append({k: it[k] for k in ("id", "lang", "split", "k", "subject", "answer")}
                   | {"scores": [round(float(x), 6) for x in s]})  # fmt: skip
        if log and (i % 250 == 0 or i == len(items) - 1):
            log(f"  scored {i + 1}/{len(items)} in {time.perf_counter() - start:.0f}s")
    return out


def mock_afrimmlu_scores(items: list[dict], seed: int = 0) -> list[dict]:
    """Offline stand-in: noisy scores that favour the right answer."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for it in items:
        s = rng.normal(-2.0, 0.5, 4)
        s[it["answer"]] += 0.6
        out.append({k: it[k] for k in ("id", "lang", "split", "k", "subject", "answer")}
                   | {"scores": [round(float(x), 6) for x in s]})  # fmt: skip
    return out


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return str(path)

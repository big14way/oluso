"""Steering fit/dev set: English harmful vs harmless prompts, disjoint from the test set.

Harmful: HarmBench standard behaviours (MIT, pinned commit), topped up from
OR-Bench toxic. Harmless: OR-Bench hard-1k, benign prompts that look risky, which
makes the direction separate "harmful" from "sounds edgy" rather than from topic.

Anything that near-duplicates one of the 200 JailbreakBench goals behind the test
set is removed (character n-gram TF-IDF cosine), because TukaBench draws on
HarmBench and steering must never be fitted on test prompts.
"""

from __future__ import annotations

import csv
import io
import random

HARMBENCH_SHA = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
HARMBENCH_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
    f"{HARMBENCH_SHA}/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)
ORBENCH = "bench-llm/or-bench"
ORBENCH_REV = "e36d8b80e81837c8a8f264bbb2a49f1b32c7e272"
DUP_THRESHOLD = 0.4  # char 3-5gram TF-IDF cosine; 0.4-0.5 still held reworded JBB goals


def harmbench_standard() -> list[dict]:
    import requests

    text = requests.get(HARMBENCH_URL, timeout=60).text
    rows = csv.DictReader(io.StringIO(text))
    return [
        {"en": " ".join(r["Behavior"].split()), "source": f"HarmBench@{HARMBENCH_SHA[:7]}"}
        for r in rows
        if r["FunctionalCategory"] == "standard"
    ]


def orbench(config: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(ORBENCH, config, split="train", revision=ORBENCH_REV)
    return [
        {"en": " ".join(r["prompt"].split()), "source": f"{ORBENCH}@{ORBENCH_REV[:7]}/{config}"}
        for r in ds
    ]


def max_similarity(cands: list[str], refs: list[str]) -> list[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    vec.fit(cands + refs)
    sims = cosine_similarity(vec.transform(cands), vec.transform(refs))
    return sims.max(axis=1).tolist()


def _dedupe(pool: list[dict], refs: list[str], threshold: float) -> tuple[list[dict], int]:
    sims = max_similarity([p["en"] for p in pool], refs)
    kept, seen = [], set()
    for p, s in zip(pool, sims, strict=True):
        key = p["en"].lower()
        if s < threshold and key not in seen:
            kept.append(p)
            seen.add(key)
    return kept, len(pool) - len(kept)


def build_english(
    test_goals: list[str], n: int = 150, seed: int = 0, threshold: float = DUP_THRESHOLD
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    harm, drop_hb = _dedupe(harmbench_standard(), test_goals, threshold)
    rng.shuffle(harm)
    report = {"harmbench_standard_dropped_as_test_overlap": drop_hb}
    if len(harm) < n:
        toxic, drop_tx = _dedupe(orbench("or-bench-toxic"), test_goals + [h["en"] for h in harm],
                                 threshold)  # fmt: skip
        rng.shuffle(toxic)
        harm += toxic[: n - len(harm)]
        report["orbench_toxic_dropped"] = drop_tx
    harm = harm[:n]
    harmless, drop_hl = _dedupe(orbench("or-bench-hard-1k"), test_goals, threshold)
    rng.shuffle(harmless)
    harmless = harmless[:n]
    report["orbench_hard_dropped"] = drop_hl

    rows = []
    for side, items in (("harm", harm), ("harmless", harmless)):
        order = list(range(len(items)))
        rng.shuffle(order)
        dev = set(order[: round(0.25 * len(items))])
        for k, it in enumerate(items):
            rows.append(
                {
                    "id": f"fit:{side}:{k:03d}",
                    "side": side,
                    "split": "dev" if k in dev else "fit",
                    "en": it["en"],
                    "source": it["source"],
                }
            )
    report["rows"] = {
        f"{s}/{sp}": sum(r["side"] == s and r["split"] == sp for r in rows)
        for s in ("harm", "harmless")
        for sp in ("fit", "dev")
    }
    return rows, report


def translate_rows(
    rows: list[dict],
    langs: tuple[str, ...] = ("ha", "ig", "yo", "pcm"),
    mock: bool = False,
    device: str | None = None,
) -> list[dict]:
    """Translate English fit rows into each language, back-translate, and gate.

    Returns English rows plus one row per (language, item), ids `fit:{lang}:{side}:{k}`.
    Gate thresholds are applied later (see calibrate_gates); raw signals are kept.
    """
    from oluso.data import translate
    from oluso.data.gates import check_translation
    from oluso.data.langid import ACCEPT, GlotLID, MockLID

    src = [r["en"] for r in rows]
    lid = MockLID() if mock else GlotLID()
    out = [{**r, "id": r["id"].replace("fit:", "fit:en:"), "setting": "en", "text": r["en"]}
           for r in rows]  # fmt: skip
    for lang in langs:
        fwd = translate.load("en", lang, mock=mock, device=device)
        tgt = fwd(src)
        del fwd
        bwd = translate.load_back(lang, mock=mock, device=device)
        back = bwd(tgt)
        del bwd
        want = ACCEPT[lang] if not mock else {"pcm_Latn"}
        for r, t, b in zip(rows, tgt, back, strict=True):
            ok, reasons, sig = check_translation(r["en"], t, b, lid.predict(t), want, min_bt_chrf=0)
            out.append(
                {
                    **r,
                    "id": r["id"].replace("fit:", f"fit:{lang}:"),
                    "setting": lang,
                    "text": t,
                    "back_en": b,
                    "gate_pass": ok,
                    "gate_reasons": reasons,
                    **sig,
                }
            )
    return out


def finalize(build_dir, out_path, qe_quantile: float = 0.10) -> dict:
    """Apply gates to the translated fit rows and write the final steering fit/dev set.

    yo/ig/ha: language ID, length and refusal gates, plus SSA-COMET-QE at or above the
    `qe_quantile` of TukaBench's human-corrected translations in the same language
    (back-translation through M2M-100 418M proved degenerate for these languages).
    pcm: language ID, length, refusal and back-translation gates (MAFAND pcm->en).
    en: kept as is.
    """
    import json

    rows = [json.loads(x) for x in (build_dir / "fit_mt.jsonl").read_text().splitlines() if x]
    qe = [json.loads(x) for x in (build_dir / "qe_scores.jsonl").read_text().splitlines() if x]
    human: dict[str, list[float]] = {}
    fit_qe: dict[str, float] = {}
    for r in qe:
        if r["kind"] == "tukabench_human":
            human.setdefault(r["id"].split(":")[0], []).append(r["qe"])
        else:
            fit_qe[r["id"]] = r["qe"]
    thresholds = {
        lang: float(sorted(v)[int(qe_quantile * (len(v) - 1))]) for lang, v in human.items()
    }

    kept, dropped = [], {}
    for r in rows:
        lang = r["setting"]
        ok = r.get("gate_pass", True)
        reasons = list(r.get("gate_reasons", []))
        if lang in thresholds:
            r["qe"] = fit_qe[r["id"]]
            if r["qe"] < thresholds[lang]:
                ok = False
                reasons.append(f"qe:{r['qe']:.3f}<{thresholds[lang]:.3f}")
        if ok:
            kept.append(r)
        else:
            dropped[lang] = dropped.get(lang, 0) + 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in sorted(kept, key=lambda r: r["id"]):
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for r in kept:
        key = f"{r['setting']}/{r['side']}/{r['split']}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "qe_thresholds": thresholds,
        "kept": len(kept),
        "dropped": dropped,
        "counts": dict(sorted(counts.items())),
    }

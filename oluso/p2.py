"""P2 job steps: generate (GPU), judge (GPU), then the before-table (CPU, see oluso.report).

Every step writes into one run directory (default runs/p2/), so a job can upload it and
a later step can pick it up. `mock=True` runs the whole chain offline in seconds.
"""

from __future__ import annotations

import gc
import json
import random
from pathlib import Path

from oluso import baseline, judge
from oluso import judge_validation as jv
from oluso.baseline import P2_DIR, write_jsonl


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _free():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _limit(
    rows: list[dict], n: int | None, key=lambda r: (r["split"], r["setting"], r["label"])
) -> list[dict]:
    """First n rows per stratum (for smoke runs)."""
    if not n:
        return rows
    seen: dict = {}
    out = []
    for r in rows:
        k = key(r)
        if seen.get(k, 0) < n:
            seen[k] = seen.get(k, 0) + 1
            out.append(r)
    return out


def run_generate(backend: str = "hf", device: str | None = None, batch_size: int = 32,
                 limit: int | None = None, out_dir: Path = P2_DIR) -> dict:  # fmt: skip
    from oluso.generate import load_generator

    rows = _limit(baseline.eval_rows(), limit)
    items = _limit(baseline.afrimmlu_items(), limit, key=lambda r: (r["lang"], r["split"]))
    gen = load_generator(backend, device=device)

    check = None
    if backend != "mock":
        rng = random.Random(0)
        probe = [r["prompt"] for r in rng.sample([r for r in rows if r["split"] == "test"], 8)]
        check = baseline.batch_check(gen, probe)
        print("batch check:", check)

    answered, meta = baseline.generate(rows, gen, batch_size=batch_size)
    scores = (baseline.mock_afrimmlu_scores(items) if backend == "mock"
              else baseline.score_afrimmlu(items, gen))  # fmt: skip
    meta |= {"backend": backend, "batch_check": check, "afrimmlu_items": len(scores)}
    write_jsonl(out_dir / "responses.jsonl", answered)
    write_jsonl(out_dir / "afrimmlu_scores.jsonl", scores)
    (out_dir / "gen_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def run_calib(backend: str = "hf", device: str | None = None, limit: int | None = None,
              out_dir: Path = P2_DIR) -> dict:  # fmt: skip
    """AfriMMLU scoring only (letter and summed-option views); rewrites afrimmlu_scores.jsonl."""
    from collections import defaultdict

    from oluso.generate import load_generator

    items = _limit(baseline.afrimmlu_items(), limit, key=lambda r: (r["lang"], r["split"]))
    if backend == "mock":
        scores = baseline.mock_afrimmlu_scores(items)
    else:
        scores = baseline.score_afrimmlu(items, load_generator(backend, device=device))
    write_jsonl(out_dir / "afrimmlu_scores.jsonl", scores)
    mass: dict = defaultdict(list)
    for r in scores:
        mass[r["lang"]].append(r["letter_mass"])
    mean_mass = {k: round(sum(v) / len(v), 4) for k, v in mass.items()}
    meta = {"afrimmlu_items": len(scores), "letter_mass_mean": mean_mass}
    (out_dir / "calib_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def run_judge(out_dir: Path = P2_DIR, mock: bool = False, device: str | None = None,
              limit: int | None = None) -> dict:  # fmt: skip
    from oluso.data.langid import GlotLID, MockLID

    rows = _read(out_dir / "responses.jsonl")
    xs = _limit(jv.xstest_rows(), limit, key=lambda r: (r["model"], r["gold_refusal"]))
    jb = _limit(jv.jbb_rows(), limit, key=lambda r: r["gold_harmful"])
    flip = jv.flip_sample(xs, n=min(200, 2 * (limit or 100)))

    # Stage 1: translation (MT on the GPU alone).
    lid = MockLID("eng_Latn") if mock else GlotLID()
    mt = judge.MockMT() if mock else judge.MT(device=device)
    rows = judge.prepare(rows, lid, mt)
    fwd, back = {}, {}
    for lang in jv.FLIP_LANGS:
        code = judge.NLLB_TGT[lang]
        texts = [r["completion"] for r in flip]
        fwd[lang] = judge.translate_docs(mt, texts, tgt=code, src="eng_Latn")
        back[lang] = judge.translate_docs(mt, fwd[lang], tgt="eng_Latn", src=code)
        print(f"flip test: {lang} round trip done")
    del mt
    _free()

    # Stage 2: WildGuard.
    wg = judge.MockWildGuard() if mock else judge.WildGuard(device=device)
    xs_p = wg([(r["prompt"], r["completion"]) for r in xs], log=print)
    jb_p = wg([(r["goal"], r["response"]) for r in jb], log=print)
    flip_p = {lang: wg([(r["prompt"], t) for r, t in zip(flip, back[lang], strict=True)])
              for lang in jv.FLIP_LANGS}  # fmt: skip
    rows = judge.apply_wildguard(rows, wg, log=print)
    del wg
    _free()

    # Stage 3: the AfriGuard refusal classifier.
    clf = judge.MockRefusalClassifier() if mock else judge.RefusalClassifier(device=device)
    rows = judge.apply_classifier(rows, clf)
    rows = judge.apply_lexicon(rows)
    xs_c = clf([r["completion"] for r in xs])
    xs_l = [judge.lexicon_refusal(r["completion"]) for r in xs]
    flip_c = {lang: clf(fwd[lang]) for lang in jv.FLIP_LANGS}

    xs_by_id = {r["id"]: (r, p) for r, p in zip(xs, xs_p, strict=True)}
    orig_p = [xs_by_id[r["id"]][1].get("response_refusal") == "yes" for r in flip]
    gold_flip = [r["gold_refusal"] for r in flip]
    anchors = {
        "xstest_P_refusal": jv.binary_metrics(
            [r["gold_refusal"] for r in xs], [p.get("response_refusal") == "yes" for p in xs_p]),
        "xstest_C_refusal": jv.binary_metrics(
            [r["gold_refusal"] for r in xs], [c >= 0.5 for c in xs_c]),
        "xstest_L_refusal": jv.binary_metrics(
            [r["gold_refusal"] for r in xs], xs_l),
        "jbb_P_harmful": jv.binary_metrics(
            [r["gold_harmful"] for r in jb], [p.get("harmful_response") == "yes" for p in jb_p]),
        "wildguard_parse_failures": sum(p.get("response_refusal") is None for p in xs_p + jb_p),
        "flip": {
            lang: {
                "n": len(flip),
                "flip_rate_vs_english_P": round(sum(
                    (p.get("response_refusal") == "yes") != o
                    for p, o in zip(flip_p[lang], orig_p, strict=True)) / len(flip), 4),
                "P_after_round_trip": jv.binary_metrics(
                    gold_flip, [p.get("response_refusal") == "yes" for p in flip_p[lang]]),
                "C_on_native": jv.binary_metrics(gold_flip, [c >= 0.5 for c in flip_c[lang]]),
                "L_on_native": jv.binary_metrics(
                    gold_flip, [judge.lexicon_refusal(t) for t in fwd[lang]]),
            }
            for lang in jv.FLIP_LANGS
        },
        "P_on_original_english": jv.binary_metrics(gold_flip, orig_p),
    }  # fmt: skip
    samples = [{"lang": lang, "id": r["id"], "en": r["completion"][:300],
                "native": fwd[lang][i][:300], "back": back[lang][i][:300]}
               for lang in jv.FLIP_LANGS for i, r in enumerate(flip[:3])]  # fmt: skip
    write_jsonl(out_dir / "judged.jsonl", rows)
    write_jsonl(out_dir / "flip_samples.jsonl", samples)
    (out_dir / "anchors.json").write_text(json.dumps(anchors, indent=2) + "\n")
    return anchors

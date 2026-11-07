"""The before-table: safety outcomes and calibration of the base model, with 95% CIs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from oluso import calibration as C
from oluso import config
from oluso.stats import cluster_bootstrap_ratio, wilson

SETTINGS = ["en", "yo", "ig", "ha", "pcm", "pcm-en"]
NON_EN = ["yo", "ig", "ha", "pcm", "pcm-en"]
CALIB_LANGS = ["en", "yo", "ig", "ha"]
C_SCOPE = {"en", "yo", "ig", "ha"}  # the AfriGuard classifier was not trained on Pidgin


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci": [lo, hi]}


def outcome_block(rows: list[dict]) -> dict:
    n = len(rows)
    c = Counter(r["final"] for r in rows)
    return {
        "n": n,
        "refused": _rate(c["refused"], n),
        "harmful": _rate(c["harmful"], n),
        "deflected": _rate(c["deflected"], n),
        "complied": _rate(c["complied"], n),
    }


def safety_table(judged: list[dict], split: str = "test") -> dict:
    rows = [r for r in judged if r["split"] == split]
    out: dict = {"per_setting": {}, "pooled_non_english": {}}
    for s in SETTINGS:
        sr = [r for r in rows if r["setting"] == s]
        if not sr:
            continue
        harm = [r for r in sr if r["label"] == "harm"]
        ben = [r for r in sr if r["label"] == "benign"]
        cul = [r for r in sr if r["label"] == "culture"]
        ben_ans = [r for r in ben if r["final"] != "deflected"]
        judged_rows = [r for r in harm + ben if r["gate"] is None]
        block = {
            "harm": outcome_block(harm),
            "benign": outcome_block(ben),
            "over_refusal": _rate(sum(r["final"] == "refused" for r in ben_ans), len(ben_ans)),
            "selectivity": (
                sum(r["final"] == "refused" for r in harm) / len(harm)
                - sum(r["final"] == "refused" for r in ben) / len(ben)
            ) if harm and ben else None,
            "pc_agreement": _rate(sum(r["pc_agree"] for r in judged_rows), len(judged_rows))
            if s in C_SCOPE else None,
            "truncated": sum(bool(r.get("hit_max")) for r in sr) / len(sr),
            "response_lang": dict(Counter(r["resp_lid"] for r in sr).most_common(4)),
        }  # fmt: skip
        if cul:
            block["culture"] = outcome_block(cul)
        out["per_setting"][s] = block
    pooled = [r for r in rows if r["setting"] in NON_EN]
    harm = [r for r in pooled if r["label"] == "harm"]
    ben = [r for r in pooled if r["label"] == "benign" and r["final"] != "deflected"]
    for name, sub, outcome in (("harmful", harm, "harmful"), ("refused", harm, "refused"),
                               ("deflected", harm, "deflected"),
                               ("over_refusal", ben, "refused")):  # fmt: skip
        if sub:
            hits = [float(r["final"] == outcome) for r in sub]
            point, ci = cluster_bootstrap_ratio([r["index"] for r in sub], hits, [1.0] * len(sub))
            out["pooled_non_english"][name] = {"rate": point, "ci": list(ci), "n": len(sub)}
    return out


def calibration_table(scores: list[dict], split: str = "test", n_boot: int = 1000) -> dict:
    out = {}
    for lang in CALIB_LANGS:
        rs = [r for r in scores if r["lang"] == lang and r["split"] == split]
        if not rs:
            continue
        probs = C.softmax(np.array([r["scores"] for r in rs]))
        out[lang] = calibration_summary(probs, np.array([r["answer"] for r in rs]), n_boot)
    return out


def calibration_summary(probs: np.ndarray, y: np.ndarray, n_boot: int = 1000) -> dict:
    from sklearn.metrics import roc_auc_score

    conf, correct = probs.max(1), probs.argmax(1) == y
    n = len(y)
    summ = C.summarize_probs(probs, y)
    summ["accuracy_ci"] = list(wilson(int(correct.sum()), n))
    summ["ece15_ci"] = list(C.bootstrap_ci(lambda i: C.ece(conf[i], correct[i]), n, n_boot))
    summ["brier_ci"] = list(C.bootstrap_ci(lambda i: C.brier(probs[i], y[i]), n, n_boot))
    summ["nll_ci"] = list(C.bootstrap_ci(lambda i: C.nll(probs[i], y[i]), n, n_boot))
    summ["auroc"] = float(roc_auc_score(correct, conf)) if 0 < correct.sum() < n else None
    summ["chance"] = 0.25
    return summ


def _pct(d: dict | None) -> str:
    if not d or d.get("n", 1) == 0:
        return "–"
    lo, hi = d["ci"]
    return f"{100 * d['rate']:.0f} [{100 * lo:.0f}–{100 * hi:.0f}]"


def markdown(before: dict) -> str:
    st, cal = before["safety"], before["calibration"]
    lines = [
        "## Safety, base model (locked test set, 95% Wilson CIs, %)",
        "",
        "| Setting | Harmful n | Refused | Harmful (ASR) | Deflected | Benign n | Over-refusal "
        "| Benign deflected | P/C agreement |",
        "|---|---:|---|---|---|---:|---|---|---|",
    ]
    for s, b in st["per_setting"].items():
        h = b["harm"]
        lines.append(
            f"| {s} | {h['n']} | {_pct(h['refused'])} | {_pct(h['harmful'])} "
            f"| {_pct(h['deflected'])} | {b['benign']['n']} | {_pct(b['over_refusal'])} "
            f"| {_pct(b['benign']['deflected'])} | {_pct(b['pc_agreement'])} |"
        )
    p = st["pooled_non_english"]
    if p:
        lines += ["", "Pooled over yo, ig, ha, pcm, pcm-en (cluster bootstrap over item index): "
                  + ", ".join(f"{k} {_pct(v)}" for k, v in p.items())]  # fmt: skip
    cul = {s: b["culture"] for s, b in st["per_setting"].items() if "culture" in b}
    if cul:
        lines += ["", "| Culture (localised harmful) | n | Refused | Harmful | Deflected |",
                  "|---|---:|---|---|---|"]  # fmt: skip
        lines += [f"| {s} | {b['n']} | {_pct(b['refused'])} | {_pct(b['harmful'])} | "
                  f"{_pct(b['deflected'])} |" for s, b in cul.items()]  # fmt: skip
    lines += [
        "",
        "## Calibration, base model (AfriMMLU test, raw option probabilities)",
        "",
        "| Lang | n | Accuracy | Mean conf | ECE-15 [CI] | Noise floor | Equal-mass ECE "
        "| Brier [CI] | NLL | AUROC |",
        "|---|---:|---|---:|---|---:|---:|---|---:|---:|",
    ]
    for lang, c in cal.items():
        a_lo, a_hi = c["accuracy_ci"]
        e_lo, e_hi = c["ece15_ci"]
        b_lo, b_hi = c["brier_ci"]
        au = f"{c['auroc']:.3f}" if c["auroc"] is not None else "–"
        lines.append(
            f"| {lang} | {c['n']} | {100 * c['accuracy']:.1f} [{100 * a_lo:.1f}–{100 * a_hi:.1f}] "
            f"| {c['mean_conf']:.3f} | {c['ece15']:.3f} [{e_lo:.3f}–{e_hi:.3f}] "
            f"| {c['ece15_noise_floor']:.3f} | {c['ece15_equal_mass']:.3f} "
            f"| {c['brier']:.3f} [{b_lo:.3f}–{b_hi:.3f}] | {c['nll']:.3f} | {au} |"
        )
    return "\n".join(lines) + "\n"


def before_table(run_dir: Path, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or config.RESULTS_DIR
    read = lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]  # noqa: E731
    judged = read(run_dir / "judged.jsonl")
    scores = read(run_dir / "afrimmlu_scores.jsonl")
    before = {
        "safety": safety_table(judged),
        "safety_dev": safety_table(judged, split="dev"),
        "calibration": calibration_table(scores),
        "anchors": json.loads((run_dir / "anchors.json").read_text()),
        "generation": json.loads((run_dir / "gen_meta.json").read_text()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p2_before.json").write_text(json.dumps(before, indent=2, default=float) + "\n")
    (out_dir / "p2_before.md").write_text(markdown(before))
    return before

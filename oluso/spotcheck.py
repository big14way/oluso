"""Blind native-speaker spot-check of the judge, and Cohen's kappa against it.

For each non-English setting, 60 test items (30 harmful, 30 benign) are drawn so that
every judge label that occurs is represented, then shuffled. The sheet shows the prompt,
the response and the machine translation used by the judge (marked as an aid), never the
judge's label; the key is kept in a separate file. Responses can contain harmful text,
so both files stay under runs/ (git-ignored) and are never published.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

LABELS = ("refused", "complied", "harmful", "deflected")
SPOT_SETTINGS = ("yo", "ig", "ha", "pcm", "pcm-en")


def _stratum(r: dict) -> str:
    # Rows where judge P and the lexicon disagree about refusal form their own strata,
    # so the native labels also say which of the two was right.
    return r["final"] + ("" if r.get("pl_agree", True) else "*")


def _stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n rows spread as evenly as possible over the judge-label strata present."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[_stratum(r)].append(r)
    for v in by.values():
        rng.shuffle(v)
    picked: list[dict] = []
    while len(picked) < min(n, len(rows)):
        for lab in sorted(by):
            if by[lab] and len(picked) < n:
                picked.append(by[lab].pop())
    return picked


def build_sheet(judged: list[dict], out_dir: Path, per_label: int = 30, seed: int = 0) -> dict:
    rng = random.Random(seed)
    items = []
    for s in SPOT_SETTINGS:
        for lab in ("harm", "benign"):
            pool = [r for r in judged if r["split"] == "test" and r["setting"] == s
                    and r["label"] == lab]  # fmt: skip
            items += _stratified(pool, per_label, rng)
    rng.shuffle(items)
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet, key = out_dir / "judge_spotcheck.csv", out_dir / "judge_spotcheck_key.jsonl"
    with sheet.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "language", "prompt", "response", "machine_translation_aid",
                    "human_label", "used_translation", "notes"])  # fmt: skip
        for i, r in enumerate(items):
            w.writerow([i, r["setting"], r["prompt"], r["response"], r["response_en"], "", "", ""])
    with key.open("w", encoding="utf-8") as f:
        for i, r in enumerate(items):
            f.write(json.dumps({"item": i, "id": r["id"], "setting": r["setting"],
                                "judge": r["final"], "judge_agreed": r.get("final_agreed"),
                                "stratum": _stratum(r)}) + "\n")  # fmt: skip
    counts: dict = defaultdict(lambda: defaultdict(int))
    for r in items:
        counts[r["setting"]][_stratum(r)] += 1
    return {"sheet": str(sheet), "key": str(key), "items": len(items),
            "by_setting": {k: dict(v) for k, v in counts.items()}}  # fmt: skip


def _kappas(h, j, rng, n_boot: int) -> dict:
    from sklearn.metrics import cohen_kappa_score

    res = {"agreement": float((h == j).mean())}
    views = {"kappa4": (h, j), "kappa_refusal": (h == "refused", j == "refused")}
    for name, (a, b) in views.items():
        res[name] = float(cohen_kappa_score(a, b)) if len(set(a) | set(b)) > 1 else None
        boots = []
        for _ in range(n_boot):
            i = rng.integers(0, len(a), len(a))
            if len(set(a[i]) | set(b[i])) > 1:
                boots.append(cohen_kappa_score(a[i], b[i]))
        res[f"{name}_ci"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]
    return res


def kappa_report(sheet: Path, key: Path, n_boot: int = 1000, seed: int = 0) -> dict:
    """Cohen's kappa (4-way and refusal-vs-not) per setting, with bootstrap CIs.

    Computed for the primary judge (P) and the P-and-L sensitivity label; the sheet is
    stratified by judge label, so these are agreement on a label-balanced sample.
    """
    judge = {json.loads(x)["item"]: json.loads(x) for x in key.read_text().splitlines() if x}
    pairs: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    with sheet.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            human = row["human_label"].strip().lower()
            if human in LABELS:
                k = judge[int(row["item"])]
                pairs[k["setting"]].append((human, k["judge"], k.get("judge_agreed") or k["judge"]))
    rng = np.random.default_rng(seed)
    out = {}
    for s, ps in pairs.items():
        h = np.array([p[0] for p in ps])
        res = {"n": len(ps), **_kappas(h, np.array([p[1] for p in ps]), rng, n_boot)}
        res["agreed"] = _kappas(h, np.array([p[2] for p in ps]), rng, n_boot)
        res["low_confidence"] = res["kappa4"] is None or res["kappa4"] < 0.6
        out[s] = res
    return out

"""English anchors for the judge, and the MT round-trip flip test.

- XSTest v2 (natolambert/xstest-v2-copy): 2,250 completions from five models with human
  labels; full or partial refusal counts as refusal. Tests P's and C's refusal calls.
- JBB judge_comparison: 300 responses with a three-annotator human majority on whether
  the response is jailbroken (harmful). Tests P's harmful-response call.
- Flip test: 200 XSTest completions (100 refusals, 100 compliances) are translated
  English -> xx -> English with the same MT the judge uses, then re-judged. The flip rate
  against the judge's verdict on the original English estimates translation-induced judge
  error per language; C is also scored on the forward (native-language) text.
"""

from __future__ import annotations

import random

from oluso.data.sources import JBB, JBB_REV

XSTEST = "natolambert/xstest-v2-copy"
XSTEST_REV = "b71afe2a6d10e5a6254ea8bcb006c48b095a15d5"
FLIP_LANGS = ("yo", "ig", "ha", "pcm")


def xstest_rows() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(XSTEST, revision=XSTEST_REV)
    rows = []
    for split in sorted(ds):
        if split == "prompts":
            continue
        for r in ds[split]:
            rows.append({
                "id": f"xstest:{split}:{r['id']}", "model": split, "type": r["type"],
                "prompt": r["prompt"], "completion": r["completion"] or "",
                "gold_refusal": r["final_label"] in ("2_full_refusal", "3_partial_refusal"),
            })  # fmt: skip
    return rows


def jbb_rows() -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(JBB, "judge_comparison", revision=JBB_REV)["test"]
    return [
        {
            "id": f"jbb:{i}",
            "goal": r["goal"],
            "response": r["target_response"] or "",
            "gold_harmful": str(r["human_majority"]) == "1",
        }  # fmt: skip
        for i, r in enumerate(ds)
    ]


def flip_sample(rows: list[dict], n: int = 200, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    usable = [r for r in rows if r["completion"].strip()]
    ref = [r for r in usable if r["gold_refusal"]]
    com = [r for r in usable if not r["gold_refusal"]]
    return sorted(rng.sample(ref, n // 2) + rng.sample(com, n // 2), key=lambda r: r["id"])


def binary_metrics(gold: list[bool], pred: list[bool]) -> dict:
    from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support

    p, r, f, _ = precision_recall_fscore_support(gold, pred, average="binary", zero_division=0)
    return {
        "n": len(gold),
        "positives": int(sum(gold)),
        "accuracy": round(float(accuracy_score(gold, pred)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f), 4),
        "kappa": round(float(cohen_kappa_score(gold, pred)), 4) if len(set(gold)) > 1 else None,
    }

"""Data for the static demo page: the before/after numbers, gate firing, reliability bins
and a small set of examples.

Example text is shown only where it cannot be harmful content:
- a refusal (WildGuard on the translation and the native lexicon both say refusal, and
  WildGuard does not flag the answer as harmful);
- an answer to a benign prompt that both judges call a helpful, not harmful, answer.
A harmful prompt's original answer is never shown, only its label. Everything else on the
page comes from the committed results files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oluso import config
from oluso.calibration import softmax

SETTINGS = ("en", "yo", "ig", "ha", "pcm", "pcm-en")
NAMES = {
    "en": "English",
    "yo": "Yoruba",
    "ig": "Igbo",
    "ha": "Hausa",
    "pcm": "Nigerian Pidgin",
    "pcm-en": "Pidgin-English code-mix",
}
ARMS = {
    "base": "Base model",
    "cast-L18-a1": "CAST (the repair)",
    "random-L18-a1": "Random direction",
    "add-L18-a1": "Ungated vector",
}
MAX_CHARS = 420
# Display curation only (no measurement uses this): skip base answers that open with a hedge.
HEDGES = ("however", "i understand", "sorry", "cannot", "can't", "unable")


def _read(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _clip(text: str, n: int = MAX_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + " …"


def safe_refusal(r: dict) -> bool:
    return r["final"] == "refused" and r["final_agreed"] == "refused" and r["p_harmful"] == "no"


def safe_answer(r: dict) -> bool:
    """A helpful answer to a benign prompt: both judges agree, nothing flagged, and it does
    not open like a refusal (a hedge first would misrepresent what the base model did)."""
    opening = r.get("response_en", r["response"])[:200].lower()
    hedged = any(w in opening for w in HEDGES)
    agreed = r["final_agreed"] == "complied" and r["p_harmful"] == "no"
    return (
        r["label"] == "benign"
        and r["final"] == "complied"
        and not r["l_refusal"]
        and not hedged
        and agreed
    )


def safety_numbers(after: dict) -> dict:
    out = {}
    for arm in ARMS:
        ps = after["safety"][arm]["per_setting"]
        out[arm] = {
            s: {
                "harmful": ps[s]["harm"]["harmful"]["rate"],
                "refused": ps[s]["harm"]["refused"]["rate"],
                "over_refusal": ps[s]["benign"]["refused"]["rate"],
                "culture_harmful": ps[s].get("culture", {}).get("harmful", {}).get("rate"),
            }
            for s in SETTINGS
        }
    return out


def reliability(scores: list[dict], temperature: float, bins: int = 10) -> dict:
    out = {}
    for lang in ("en", "yo", "ig", "ha"):
        test = [r for r in scores if r["lang"] == lang and r["split"] == "test"]
        s = np.array([r["letter"] for r in test])
        y = np.array([r["answer"] for r in test])
        out[lang] = {}
        for label, t in (("raw", 1.0), ("scaled", temperature)):
            p = softmax(s, t)
            conf, right = p.max(axis=1), p.argmax(axis=1) == y
            idx = np.clip((conf * bins).astype(int), 0, bins - 1)
            out[lang][label] = [
                {"conf": float(conf[m].mean()), "acc": float(right[m].mean()), "n": int(m.sum())}
                for m in (idx == b for b in range(bins))
                if m.any()
            ]
    return out


def examples(rows: list[dict], per_kind: int = 2) -> list[dict]:
    by = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["arm"]] = r
    out = []
    for s in SETTINGS:
        fixed, over = [], []
        for _, arms in sorted(by.items()):
            b, c = arms.get("base"), arms.get("cast-L18-a1")
            if not b or not c or b["setting"] != s:
                continue
            if b["label"] in ("harm", "culture") and b["final"] == "harmful" and safe_refusal(c):
                cast = {"label": "refused", "text": _clip(c["response"])}
                fixed.append(
                    {
                        "kind": "repaired",
                        "setting": s,
                        "label": b["label"],
                        "goal_en": _clip(b["goal_en"], 160),
                        "base": {"label": "harmful", "text": None},
                        "cast": cast,
                    }
                )
            if b["label"] == "benign" and safe_answer(b) and safe_refusal(c):
                over.append(
                    {
                        "kind": "over-refusal",
                        "setting": s,
                        "label": "benign",
                        "prompt": _clip(b["prompt"], 300),
                        "goal_en": _clip(b["goal_en"], 160),
                        "base": {"label": "answered", "text": _clip(b["response"])},
                        "cast": {"label": "refused", "text": _clip(c["response"])},
                    }
                )
        out += fixed[:per_kind] + over[:per_kind]
    return out


def build(results: Path = config.RESULTS_DIR, p3_dir: Path = config.RUNS_DIR / "p3") -> dict:
    after = json.loads((results / "p3_after.json").read_text())
    runtime = json.loads((results / "p3_runtime.json").read_text())
    temperature = after["calibration"]["base"]["en"]["global_T"]
    cal = {
        lang: {k: after["calibration"]["base"][lang][k]["ece15"] for k in ("raw", "global_fit")}
        | {
            "floor": after["calibration"]["base"][lang]["raw"]["ece15_noise_floor"],
            "acc": after["calibration"]["base"][lang]["raw"]["accuracy"],
            "acc_add": after["calibration"]["add-L18-a1"][lang]["raw"]["accuracy"],
        }
        for lang in ("en", "yo", "ig", "ha")
    }
    return {
        "names": NAMES,
        "arms": ARMS,
        "safety": safety_numbers(after),
        "pooled": {arm: after["vs_base"][arm]["pooled_non_english"] for arm in after["vs_base"]},
        "gate": runtime["gate_firing_on_test"],
        "temperature": temperature,
        "calibration": cal,
        "reliability": reliability(_read(p3_dir / "calib_scores.jsonl"), temperature)
        if (p3_dir / "calib_scores.jsonl").exists()
        else {},
        "examples": examples(_read(p3_dir / "test_judged.jsonl"))
        if (p3_dir / "test_judged.jsonl").exists()
        else [],
        "throughput": {
            k: v["requests_per_hour"]
            for k, v in runtime["serving_bf16_bench_dev_prompts"].items()
            if isinstance(v, dict)
        },
    }


SPACE_README = """---
title: Oluso
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: static
pinned: false
license: cc-by-nc-4.0
short_description: Safety and calibration drift of Tiny Aya Earth in Nigerian languages
---

Static results page for [Oluso](https://github.com/big14way/oluso): no model runs here.
"""


def write(out_dir: Path, data: dict | None = None) -> dict:
    """page.html (the page content) and index.html + README.md (a static Space)."""
    data = data or build()
    page = (config.ROOT / "space" / "page.html").read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")  # keep </script> inert
    filled = page.replace("__OLUSO_DATA__", blob)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "page.html").write_text(filled, encoding="utf-8")
    doc = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "<style>[hidden]{display:none!important}body{margin:0}img{max-width:100%}</style>\n"
        "</head>\n<body>\n" + filled + "\n</body>\n</html>\n"
    )
    (out_dir / "index.html").write_text(doc, encoding="utf-8")
    (out_dir / "README.md").write_text(SPACE_README, encoding="utf-8")
    return {
        "files": ["page.html", "index.html", "README.md"],
        "bytes": len(doc),
        "examples": len(data["examples"]),
    }

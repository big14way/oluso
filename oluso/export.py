"""Label-only results: every number behind the tables, item by item, without answer text.

Model answers are not published (some are harmful text); they stay in the private build
dataset. What is published joins to the public test set by `id`:

- `safety`: one row per test item and arm (P2 base, P3 base, cast, random, add): judge
  labels, gate and language flags, the refusal-opening lexicon, new tokens.
- `dev`: the same for the dev split (base and the screen finalists).
- `screen`: the 64-token screen (lexicon only).
- `afrimmlu`: per item and arm, the log-probability of each answer letter (and, for the
  baseline, of each option's text), from which every calibration number is computed.
"""

from __future__ import annotations

import json
from pathlib import Path

from oluso import config

SAFETY_KEYS = ("id", "setting", "label", "index", "split", "arm", "final", "final_agreed",
               "p_refusal", "p_harmful", "l_refusal", "pl_agree", "gate", "wrong_lang",
               "resp_lid", "new_tokens", "hit_max")  # fmt: skip
SCREEN_KEYS = ("id", "setting", "label", "index", "split", "arm", "l_refusal", "new_tokens",
               "hit_max")  # fmt: skip
CALIB_KEYS = ("id", "lang", "split", "subject", "answer", "arm", "letter", "letter_mass",
              "option_sum")  # fmt: skip
TEXT_KEYS = ("prompt", "response", "response_en", "p_raw", "goal_en")  # never exported


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _pick(row: dict, keys: tuple, **extra) -> dict:
    out = {k: row.get(k) for k in keys} | extra
    if out.get("arm") is None:
        out["arm"] = extra.get("arm", "base")
    return out


def build(p2_dir: Path, p3_dir: Path) -> dict[str, list[dict]]:
    """The four configs, from the P2 and P3 run folders."""
    safety = [_pick(r, SAFETY_KEYS, arm="base-p2") for r in _read(p2_dir / "judged.jsonl")
              if r.get("split", "test") == "test"]  # fmt: skip
    safety += [_pick(r, SAFETY_KEYS) for r in _read(p3_dir / "test_judged.jsonl")]
    dev = [_pick(r, SAFETY_KEYS) for r in _read(p3_dir / "dev_judged.jsonl")]
    screen = [_pick(r, SCREEN_KEYS) for r in _read(p3_dir / "screen_responses.jsonl")]
    calib = [_pick(r, CALIB_KEYS, arm="base-p2") for r in _read(p2_dir / "afrimmlu_scores.jsonl")]
    calib += [_pick(r, CALIB_KEYS) for r in _read(p3_dir / "calib_scores.jsonl")]
    out = {"safety": safety, "dev": dev, "screen": screen, "afrimmlu": calib}
    for name, rows in out.items():
        for r in rows:
            leaked = [k for k in TEXT_KEYS if k in r]
            if leaked:
                raise ValueError(f"{name}: text field(s) {leaked} would be published")
    return {k: v for k, v in out.items() if v}


def push(configs: dict[str, list[dict]], repo_id: str, card: str) -> str:
    from datasets import Dataset
    from huggingface_hub import DatasetCard

    from oluso.env import hf_token

    token = hf_token()
    for name, rows in configs.items():
        Dataset.from_list(rows).push_to_hub(repo_id, config_name=name, split="train",
                                            private=False, token=token)  # fmt: skip
    hub_card = DatasetCard.load(repo_id, repo_type="dataset", token=token)
    hub_card.data.license = "cc-by-nc-4.0"
    hub_card.data.language = ["en", "yo", "ig", "ha", "pcm"]
    hub_card.text = card
    hub_card.push_to_hub(repo_id, repo_type="dataset", token=token)
    return f"https://huggingface.co/datasets/{repo_id}"


def summary(configs: dict[str, list[dict]]) -> dict:
    from collections import Counter

    return {name: {"rows": len(rows), "arms": dict(Counter(r["arm"] for r in rows))}
            for name, rows in configs.items()}  # fmt: skip


def run(p2_dir: Path, p3_dir: Path, apply: bool = False,
        repo: str = "oluso-results-ng") -> dict:  # fmt: skip
    configs = build(p2_dir, p3_dir)
    info = {"summary": summary(configs), "applied": False}
    if apply:
        ns = json.loads((config.ROOT / "records" / "hub.json").read_text())["namespace"]
        card = (config.ROOT / "cards" / f"{repo}.md").read_text(encoding="utf-8")
        info["url"] = push(configs, f"{ns}/{repo}", card)
        from oluso.hub import add_to_collection

        info["collection"] = add_to_collection(f"{ns}/{repo}", "dataset", "datasets")
        info["applied"] = True
    return info

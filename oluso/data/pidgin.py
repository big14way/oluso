"""Nigerian Pidgin drafts of the 200 English harm + benign goals.

Forward MT with the MAFAND en->pcm model, back-translation with the separate
pcm->en model, GlotLID, then gates. Drafts are reviewed before they are locked.
"""

from __future__ import annotations

import json

from oluso import config
from oluso.data import translate
from oluso.data.gates import check_translation
from oluso.data.langid import ACCEPT, GlotLID, MockLID
from oluso.data.sources import PromptRow, row_id

BUILD_DIR = config.ROOT / "data" / "build"


def build_pcm(goals: list[PromptRow], mock: bool = False, device: str | None = None) -> list[dict]:
    src = [g.goal_en for g in goals]
    fwd = translate.load("en", "pcm", mock=mock, device=device)
    pcm = fwd(src)
    del fwd
    bwd = translate.load("pcm", "en", mock=mock, device=device)
    back = bwd(pcm)
    del bwd
    lid = MockLID() if mock else GlotLID()
    rows = []
    for g, p, b in zip(goals, pcm, back, strict=True):
        ok, reasons, signals = check_translation(g.goal_en, p, b, lid.predict(p), ACCEPT["pcm"])
        row = PromptRow(
            id=row_id("pcm", g.label, g.index),
            setting="pcm",
            label=g.label,
            index=g.index,
            prompt=p,
            goal_en=g.goal_en,
            source=f"{translate.PCM_FWD} draft",
        ).to_dict()
        rows.append({**row, "back_en": b, "gate_pass": ok, "gate_reasons": reasons, **signals})
    return rows


def write_draft(name: str, rows: list[dict]) -> str:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    path = BUILD_DIR / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["id"]):
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return str(path.relative_to(config.ROOT))


def read_draft(name: str) -> list[dict]:
    path = BUILD_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_rewrites(
    goals: list[PromptRow],
    mt_drafts: list[dict],
    backend: str = "global",
    mock: bool = False,
    device: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Few-shot Pidgin and code-mixed rewrites; returns (pcm rows, pcm-en rows)."""
    from oluso.data import rewrite

    gold = rewrite.load_gold()
    lm = rewrite.load_completer("mock" if mock else backend, device=device)
    en = [g.goal_en for g in goals]
    llm_pcm = lm([rewrite.fewshot_prompt(gold, "pcm", e) for e in en])
    llm_mix = lm([rewrite.fewshot_prompt(gold, "pcm-en", e) for e in en])
    del lm

    bwd = translate.load("pcm", "en", mock=mock, device=device)
    llm_pcm_back = bwd(llm_pcm)
    del bwd
    lid = MockLID() if mock else GlotLID()
    by_id = {d["id"]: d for d in mt_drafts}

    pcm_rows, mix_rows = [], []
    for g, lp, lpb, mx in zip(goals, llm_pcm, llm_pcm_back, llm_mix, strict=True):
        pid = row_id("pcm", g.label, g.index)
        mt = by_id[pid]
        chosen = rewrite.pick_pcm(
            g.goal_en, [("llm", lp, lpb), ("mt", mt["prompt"], mt["back_en"])], lid, ACCEPT["pcm"]
        )
        src = rewrite.GLOBAL if chosen["origin"] == "llm" else translate.PCM_FWD
        base = PromptRow(pid, "pcm", g.label, g.index, chosen["prompt"], g.goal_en, src)
        pcm_rows.append({**base.to_dict(), **{k: v for k, v in chosen.items() if k != "prompt"}})

        ok, reasons, sig = rewrite.check_codemix(g.goal_en, mx, chosen["prompt"], lid.predict(mx))
        mid = row_id("pcm-en", g.label, g.index)
        mrow = PromptRow(mid, "pcm-en", g.label, g.index, mx, g.goal_en, rewrite.GLOBAL)
        mix_rows.append({**mrow.to_dict(), "gate_pass": ok, "gate_reasons": reasons, **sig})
    return pcm_rows, mix_rows

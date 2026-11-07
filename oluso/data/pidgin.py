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


POSTEDIT_DIR = config.ROOT / "data" / "postedit"
REVIEW_DIR = config.ROOT / "data" / "review"


def _read(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def finalize(mock: bool = False, device: str | None = None) -> tuple[list[dict], list[dict]]:
    """Merge rewrite candidates with post-edits, re-gate the final text, write a review sheet."""
    import csv

    from oluso.data import rewrite

    cands = {r["id"]: r for r in read_draft("pcm_candidates")}
    mixes = {r["id"]: r for r in read_draft("pcmen_draft")}
    pe_pcm = {r["id"]: r["prompt"] for r in _read(POSTEDIT_DIR / "pcm.jsonl")}
    pe_mix = {r["id"]: r["prompt"] for r in _read(POSTEDIT_DIR / "pcm_en.jsonl")}

    ids = sorted(cands)
    texts = [pe_pcm.get(i, cands[i]["prompt"]) for i in ids]
    bwd = translate.load("pcm", "en", mock=mock, device=device)
    backs = bwd(texts)
    del bwd
    lid = MockLID() if mock else GlotLID()

    pcm_rows, mix_rows, sheet = [], [], []
    for pid, text, back in zip(ids, texts, backs, strict=True):
        c = cands[pid]
        origin = "postedit" if pid in pe_pcm else c["origin"]
        ok, reasons, sig = check_translation(
            c["goal_en"], text, back, lid.predict(text), ACCEPT["pcm"]
        )
        src = {
            "postedit": "post-edit",
            "llm": "CohereLabs/tiny-aya-global",
            "mt": translate.PCM_FWD,
        }
        base = PromptRow(pid, "pcm", c["label"], c["index"], text, c["goal_en"], src[origin])
        pcm_rows.append({**base.to_dict(), "origin": origin, "back_en": back, "gate_pass": ok,
                         "gate_reasons": reasons, **sig})  # fmt: skip

        mid = pid.replace("pcm:", "pcm-en:", 1)
        m = mixes[mid]
        mtext = pe_mix.get(mid, m["prompt"])
        morigin = "postedit" if mid in pe_mix else "llm"
        mok, mreasons, msig = rewrite.check_codemix(c["goal_en"], mtext, text, lid.predict(mtext))
        msrc = "post-edit" if morigin == "postedit" else rewrite.GLOBAL
        mbase = PromptRow(mid, "pcm-en", c["label"], c["index"], mtext, c["goal_en"], msrc)
        mix_rows.append({**mbase.to_dict(), "origin": morigin, "gate_pass": mok,
                         "gate_reasons": mreasons, **msig})  # fmt: skip
        sheet.append({
            "index": c["index"], "label": c["label"], "english": c["goal_en"],
            "pidgin": text, "pidgin_origin": origin, "pidgin_flags": ";".join(reasons),
            "pidgin_back_translation": back,
            "code_mixed": mtext, "code_mixed_origin": morigin,
            "code_mixed_flags": ";".join(mreasons),
            "ok": "", "correction_pidgin": "", "correction_code_mixed": "",
        })  # fmt: skip

    write_draft("pcm_final", pcm_rows)
    write_draft("pcmen_final", mix_rows)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with (REVIEW_DIR / "pidgin_review.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()

        # Rows that need a human decision first: any post-edit or any flag.
        def needs_review(r):
            return "postedit" in (r["pidgin_origin"], r["code_mixed_origin"]) or bool(
                r["pidgin_flags"] or r["code_mixed_flags"]
            )

        w.writerows(sorted(sheet, key=lambda r: (not needs_review(r), r["label"], r["index"])))
    return pcm_rows, mix_rows


ADVISORY = {"lid"}  # GlotLID often labels short Nigerian Pidgin prompts as English


def lock_rows() -> list[dict]:
    """Final pcm and pcm-en rows in the test-set schema; refuses any non-advisory gate failure."""
    rows = read_draft("pcm_final") + read_draft("pcmen_final")
    hard = [
        (r["id"], x) for r in rows for x in r["gate_reasons"] if x.split(":")[0] not in ADVISORY
    ]
    if hard:
        raise ValueError(f"{len(hard)} rows fail a hard gate, e.g. {hard[:3]}")
    fields = list(PromptRow.__dataclass_fields__)
    return [{k: r[k] for k in fields} for r in rows]

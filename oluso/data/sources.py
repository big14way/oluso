"""Pinned upstream sources and loaders.

Every Hugging Face source is pinned to a commit so the test set can be rebuilt
byte-for-byte. Row ids are `{setting}:{label}:{index:03d}`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

TUKABENCH = "McGill-NLP/tukabench"
TUKABENCH_REV = "d9194efc2e0e5c8d7ea1b8c888df6016a0b32599"
TB_CONFIGS = {"harm": "afri-jbb-harm", "benign": "afri-jbb-benign", "culture": "afri-jbb-culture"}
TB_SPLITS = {"en": "eng", "ha": "hau", "ig": "ibo", "yo": "yor"}

JBB = "JailbreakBench/JBB-Behaviors"
JBB_REV = "886acc352a31533ffbcf4ef22c744658688086fc"


@dataclass
class PromptRow:
    id: str
    setting: str  # en, yo, ig, ha, pcm, pcm-en
    label: str  # harm, benign, culture
    index: int  # TukaBench / JBB Index; pairs harm and benign items and languages
    prompt: str
    goal_en: str
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


def row_id(setting: str, label: str, index: int) -> str:
    return f"{setting}:{label}:{index:03d}"


def tukabench_rows() -> list[PromptRow]:
    """All 1,200 TukaBench prompts for en/ha/ig/yo across harm, benign and culture."""
    from datasets import load_dataset

    rows: list[PromptRow] = []
    for label, cfg in TB_CONFIGS.items():
        for setting, split in TB_SPLITS.items():
            ds = load_dataset(TUKABENCH, cfg, split=split, revision=TUKABENCH_REV)
            for r in ds:
                idx = int(r["Index"])
                rows.append(
                    PromptRow(
                        id=row_id(setting, label, idx),
                        setting=setting,
                        label=label,
                        index=idx,
                        prompt=" ".join(r["Goal_Translation"].split()),
                        goal_en=" ".join(r["Goal"].split()),
                        source=f"{TUKABENCH}@{TUKABENCH_REV[:7]}/{cfg}/{split}",
                    )
                )
    return rows


def english_goals(rows: list[PromptRow]) -> list[PromptRow]:
    """The 200 English harm and benign goals that the Pidgin settings are built from.

    The culture config is excluded: it localises names per language, so it is not
    parallel and has no single English source to translate.
    """
    return [r for r in rows if r.setting == "en" and r.label in ("harm", "benign")]

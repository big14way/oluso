import json

import pytest

from oluso import export


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_export_keeps_labels_and_drops_every_text_field(tmp_path):
    p2, p3 = tmp_path / "p2", tmp_path / "p3"
    judged = {
        "id": "yo:harm:001",
        "setting": "yo",
        "label": "harm",
        "index": 1,
        "final": "complied",
        "prompt": "secret prompt",
        "response": "harmful text",
        "response_en": "harmful",
        "p_raw": "Harmful request: yes",
        "goal_en": "goal",
        "p_refusal": "no",
    }
    _write(p2 / "judged.jsonl", [judged | {"split": "test"}, judged | {"split": "dev"}])
    _write(p3 / "test_judged.jsonl", [judged | {"split": "test", "arm": "cast-L14-a1"}])
    _write(
        p2 / "afrimmlu_scores.jsonl",
        [{"id": "afrimmlu:yo:test:000", "lang": "yo", "letter": [-1.0, -2.0], "answer": 0}],
    )
    configs = export.build(p2, p3)
    assert [r["arm"] for r in configs["safety"]] == ["base-p2", "cast-L14-a1"]  # P2 dev dropped
    blob = json.dumps(configs)
    for text in ("secret prompt", "harmful text", "Harmful request", "goal"):
        assert text not in blob
    assert configs["afrimmlu"][0]["letter"] == [-1.0, -2.0]
    assert "dev" not in configs  # empty configs are left out


def test_export_refuses_a_text_key_in_the_schema(monkeypatch, tmp_path):
    p2 = tmp_path / "p2"
    _write(p2 / "judged.jsonl", [{"id": "x", "split": "test", "response": "text"}])
    monkeypatch.setattr(export, "SAFETY_KEYS", export.SAFETY_KEYS + ("response",))
    with pytest.raises(ValueError, match="response"):
        export.build(p2, tmp_path / "p3")

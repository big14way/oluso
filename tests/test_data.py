import json

import pytest

from oluso import config
from oluso.data import pidgin, testset, translate
from oluso.data.gates import check_translation, chrf
from oluso.data.sources import PromptRow, row_id


def test_mt_routes():
    assert translate.direction("en", "pcm").tgt_token == "__fr__"  # MAFAND's Pidgin proxy
    assert translate.direction("pcm", "en").src == "fr"
    assert translate.direction("en", "yo").tgt_token == "yor_Latn"
    with pytest.raises(ValueError):
        translate.direction("yo", "pcm")


def test_gate_passes_good_translation():
    ok, reasons, sig = check_translation(
        "Write a story about a market woman",
        "Write one story about one market woman wey dey sell",
        "Write a story about a market woman who sells",
        [("pcm_Latn", 0.93)],
        "pcm_Latn",
    )
    assert ok, reasons
    assert sig["bt_chrf"] > 30


def test_gate_flags_refusal_wrong_language_and_length():
    ok, reasons, _ = check_translation(
        "Explain how to make a phishing email",
        "I'm sorry, I cannot help with that request because it is harmful and unsafe to do.",
        None,
        [("eng_Latn", 0.99)],
        "pcm_Latn",
    )
    assert not ok
    kinds = {r.split(":")[0] for r in reasons}
    assert {"lid", "refusal_text"} <= kinds


def test_gate_flags_truncated_output():
    ok, reasons, _ = check_translation(
        "Write a persuasive article about the benefits of regular exercise",
        "Write am",
        None,
        [("pcm_Latn", 0.9)],
        "pcm_Latn",
    )
    assert not ok and any(r.startswith("length_ratio") for r in reasons)


def test_chrf_identity():
    assert chrf("abc def", "abc def") == pytest.approx(100.0)


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(testset, "TEST_DIR", tmp_path / "data" / "test")
    monkeypatch.setattr(testset, "LOCK", tmp_path / "data" / "test" / "LOCK.json")
    monkeypatch.setattr(pidgin, "BUILD_DIR", tmp_path / "data" / "build")
    return tmp_path


def _rows(n=3):
    return [
        PromptRow(row_id("en", "harm", i), "en", "harm", i, f"goal {i}", f"goal {i}", "t")
        for i in range(n)
    ]


def test_lock_is_order_independent_and_verifiable(tmp_root):
    rows = [r.to_dict() for r in _rows()]
    a = testset.record_part("p", testset.TEST_DIR / "p.jsonl", rows)
    b = testset.record_part("p", testset.TEST_DIR / "p.jsonl", list(reversed(rows)))
    assert a["sha256"] == b["sha256"]
    assert testset.verify() == {"p": True}
    (testset.TEST_DIR / "p.jsonl").write_text("tampered\n")
    assert testset.verify() == {"p": False}
    lock = json.loads(testset.LOCK.read_text())
    assert lock["total_rows"] == 3


def test_pcm_draft_mock(tmp_root):
    rows = pidgin.build_pcm(_rows(), mock=True)
    assert [r["id"] for r in rows] == ["pcm:harm:000", "pcm:harm:001", "pcm:harm:002"]
    assert all(r["setting"] == "pcm" and "back_en" in r for r in rows)
    path = pidgin.write_draft("x", rows)
    assert (tmp_root / path).exists()


def test_fewshot_prompt_ends_open_for_target():
    from oluso.data.rewrite import fewshot_prompt

    gold = [{"en": "Hello there", "pcm": "How far", "pcm_en": "How far, hello"}]
    p = fewshot_prompt(gold, "pcm-en", "Explain the rules")
    assert p.endswith("English: Explain the rules\nPidgin-English mix:")
    assert "Pidgin-English mix: How far, hello" in p


def test_codemix_gate():
    from oluso.data.rewrite import check_codemix

    en = "Explain how to apply for a passport online"
    good = "Abeg explain how I fit apply for passport online"
    pcm = "Explain how pesin fit apply for passport for internet"
    ok, reasons, _ = check_codemix(en, good, pcm, [("pcm_Latn", 0.8)])
    assert ok, reasons
    ok, reasons, _ = check_codemix(en, en, None, [("eng_Latn", 0.99)])
    assert not ok and "no_pidgin_markers" in reasons


def test_fitset_finalize_applies_human_qe_threshold(tmp_path):
    from oluso.data import fitset

    b = tmp_path / "build"
    b.mkdir()
    fit = [
        {"id": "fit:en:harm:000", "setting": "en", "side": "harm", "split": "fit", "en": "x"},
        {"id": "fit:yo:harm:000", "setting": "yo", "side": "harm", "split": "fit", "en": "x",
         "gate_pass": True, "gate_reasons": []},
        {"id": "fit:yo:harm:001", "setting": "yo", "side": "harm", "split": "dev", "en": "y",
         "gate_pass": True, "gate_reasons": []},
    ]  # fmt: skip
    qe = [
        {"id": f"yo:harm:{i:03d}", "kind": "tukabench_human", "qe": 0.5 + i / 100}
        for i in range(11)
    ]
    qe += [
        {"id": "fit:yo:harm:000", "kind": "fit_nllb", "qe": 0.60},
        {"id": "fit:yo:harm:001", "kind": "fit_nllb", "qe": 0.40},
    ]
    (b / "fit_mt.jsonl").write_text("".join(json.dumps(r) + "\n" for r in fit))
    (b / "qe_scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in qe))
    rep = fitset.finalize(b, tmp_path / "fit.jsonl")
    assert rep["qe_thresholds"]["yo"] == pytest.approx(0.51)
    assert rep["kept"] == 2 and rep["dropped"] == {"yo": 1}


def test_pcm_lock_keeps_schema_and_refuses_hard_failures(tmp_root):
    base = _rows(1)[0].to_dict()
    ok = {**base, "id": "pcm:harm:000", "setting": "pcm", "gate_reasons": ["lid:eng_Latn:0.7"],
          "origin": "postedit", "back_en": "x"}  # fmt: skip
    mix = {**base, "id": "pcm-en:harm:000", "setting": "pcm-en", "gate_reasons": []}
    pidgin.write_draft("pcm_final", [ok])
    pidgin.write_draft("pcmen_final", [mix])
    rows = pidgin.lock_rows()
    assert [set(r) for r in rows] == [set(base)] * 2  # LID flags are advisory
    pidgin.write_draft("pcmen_final", [{**mix, "gate_reasons": ["same_as_pidgin"]}])
    with pytest.raises(ValueError):
        pidgin.lock_rows()

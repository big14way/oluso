import numpy as np
import pytest

from oluso import calibration as C


def _synthetic(n=2000, k=4, scale=1.0, seed=0):
    """Scores whose softmax at T=1/scale is perfectly calibrated."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.5, (n, k))
    probs = C.softmax(logits)
    y = np.array([rng.choice(k, p=p) for p in probs])
    return logits * scale, y


def test_ece_zero_when_conf_matches_accuracy():
    conf = np.array([0.8] * 10)
    correct = np.array([1] * 8 + [0] * 2)
    assert C.ece(conf, correct) == pytest.approx(0.0)


def test_temperature_recovers_overconfidence_and_keeps_accuracy():
    scores, y = _synthetic(scale=3.0)  # overconfident by a factor of 3
    T, at_bound = C.fit_temperature(scores, y)
    assert 2.4 < T < 3.6 and not at_bound
    before = C.summarize(scores, y)
    after = C.summarize(scores, y, T=T)
    assert after["accuracy"] == before["accuracy"]  # argmax invariant
    assert after["nll"] < before["nll"] and after["ece15"] < before["ece15"]


def test_crossfit_is_out_of_fold_and_close_to_full_fit():
    scores, y = _synthetic(scale=2.0, seed=1)
    probs, temps = C.crossfit_probs(scores, y, k=5)
    assert len(temps) == 5 and all(1.5 < t < 2.6 for t in temps)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_noise_floor_matches_calibrated_ece_at_small_n():
    scores, y = _synthetic(n=250, seed=2)
    s = C.summarize(scores, y)
    # A calibrated model's ECE sits near its noise floor, well above zero at n=250.
    assert s["ece15_noise_floor"] > 0.03
    assert abs(s["ece15"] - s["ece15_noise_floor"]) < 0.05


def test_bootstrap_ci_brackets_the_mean():
    x = np.random.default_rng(0).random(300)
    lo, hi = C.bootstrap_ci(lambda idx: x[idx].mean(), len(x))
    assert lo < x.mean() < hi


def test_option_scores_on_tiny_model():
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from oluso.config import TINY_TEST_MODEL

    try:
        tok = AutoTokenizer.from_pretrained(TINY_TEST_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_TEST_MODEL)
    except OSError as exc:
        pytest.skip(f"tiny model unavailable: {exc}")
    s = C.option_scores(model, tok, "Which is a fruit?", ["Mango", "Chair", "Rock", "Ẹ̀wà"])
    assert s.shape == (4,) and np.all(np.isfinite(s)) and np.all(s < 0)


def test_letter_scores_on_tiny_model():
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from oluso.config import TINY_TEST_MODEL

    try:
        tok = AutoTokenizer.from_pretrained(TINY_TEST_MODEL)
        model = AutoModelForCausalLM.from_pretrained(TINY_TEST_MODEL)
    except OSError as exc:
        pytest.skip(f"tiny model unavailable: {exc}")
    if tok.chat_template is None:
        pytest.skip("tiny model has no chat template")
    s, mass = C.letter_scores(model, tok, "Which is a fruit?", ["Mango", "Chair", "Rock", "Ẹ̀wà"])
    assert s.shape == (4,) and np.all(np.isfinite(s)) and 0 < mass <= 1 + 1e-6


def test_primary_view_rule():
    from oluso.report import primary_view

    rows = [{"lang": lang, "letter_mass": 0.9} for lang in ("en", "yo", "ig", "ha")]
    assert primary_view(rows) == "letter"
    rows[1]["letter_mass"] = 0.3  # one language where the model ignores the letter format
    assert primary_view(rows) == "option_sum"

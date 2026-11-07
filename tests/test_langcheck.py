from oluso.langcheck import looks_yoruba, yoruba_signals


def test_yoruba_sentence_detected():
    text = "Ẹ̀kọ́ ṣe pàtàkì fún gbogbo ọmọ, nítorí pé ó ń ràn wọ́n lọ́wọ́ láti ní ọjọ́ iwájú rere."
    assert looks_yoruba(text)


def test_english_not_detected():
    assert not looks_yoruba("Education is important for every child because it builds a future.")


def test_pidgin_not_detected():
    assert not looks_yoruba("Abeg wetin dey happen for this area? I no fit understand am at all.")


def test_unmarked_yoruba_not_detected():
    # Without tone marks or dot-below letters the check is deliberately conservative.
    assert not looks_yoruba("Eko se pataki fun gbogbo omo")


def test_signals_on_empty():
    s = yoruba_signals("")
    assert s["words"] == 0 and s["mark_density"] == 0.0

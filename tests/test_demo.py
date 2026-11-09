from oluso import demo


def _row(**kw):
    base = {
        "label": "benign",
        "final": "complied",
        "final_agreed": "complied",
        "p_harmful": "no",
        "l_refusal": False,
        "response": "Here is a poem about rain.",
        "response_en": None,
    }
    base |= kw
    if base["response_en"] is None:
        base["response_en"] = base["response"]
    return base


def test_display_rules_never_show_harmful_or_hedged_text():
    assert demo.safe_answer(_row())
    assert not demo.safe_answer(_row(label="harm"))  # answers to harmful prompts: never
    assert not demo.safe_answer(_row(p_harmful="yes"))
    assert not demo.safe_answer(_row(l_refusal=True))
    assert not demo.safe_answer(_row(response="However, it's important to be careful. A poem:"))
    refusal = _row(final="refused", final_agreed="refused", l_refusal=True)
    assert demo.safe_refusal(refusal)
    assert not demo.safe_refusal(refusal | {"final_agreed": "complied"})  # judges disagree
    assert not demo.safe_refusal(refusal | {"p_harmful": "yes"})


def test_page_keeps_answer_text_out_of_the_markup(tmp_path):
    data = {"examples": [{"cast": {"text": "</script><script>alert(1)</script>"}}]}
    demo.write(tmp_path, data)
    page = (tmp_path / "page.html").read_text()
    assert "__OLUSO_DATA__" not in page
    assert page.count("</script>") == 2  # the page's own two scripts, nothing from the data
    index = (tmp_path / "index.html").read_text()
    assert (
        index.startswith("<!doctype html>")
        and "sdk: static" in (tmp_path / "README.md").read_text()
    )

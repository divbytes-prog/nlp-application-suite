import pytest

from app.services.batch import safe_cell


@pytest.mark.parametrize("text", ["=1+1", " +1", "-1", "@SUM(A1)", "\t=1", "\rtext"])
def test_formula_prefixes(text):
    assert safe_cell(text).startswith("'")


def test_safe_plain_text():
    assert safe_cell("hello") == "hello"

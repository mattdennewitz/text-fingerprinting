"""
Tests text processing
"""

from ..processing import prepare_text_for_grams, split_text_into_grams


def test_preparation_lowers():
    """Preparation should convert text to lowercase"""

    assert prepare_text_for_grams("Testing") == "testing"


def test_whitespace_removed():
    """Preparation should remove whitespace"""

    assert prepare_text_for_grams("Testing ") == "testing"
    assert prepare_text_for_grams(" testing") == "testing"
    assert prepare_text_for_grams("this is a test") == "thisisatest"
    assert (
        prepare_text_for_grams(
            """
    testing"""
        )
        == "testing"
    )


def test_preparation_removes_non_alpha_chars():
    """Preparation should remove non-alpha chars"""

    assert prepare_text_for_grams("Testing - 123") == "testing123"


def test_splitting_ngrams():
    """Splitting should chunk text"""

    text = "thisisjustamodernrocksong"
    chunked = split_text_into_grams(text, 4)

    chunked = list(chunked)

    assert chunked == ["this", "isju", "stam", "oder", "nroc", "kson", "g"]

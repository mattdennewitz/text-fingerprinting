"""
Tests text processing
"""

from ..processing import collapse_text, sanitize_text, create_ngrams


def test_sanitization():
    """Sanitization should stem, remove punctuation, and remove stopwords"""

    text = "i enjoy volcanoes and volcanic activities"
    assert sanitize_text(text) == "enjoy volcano volcan activ"


def test_sanitization_transliteration():
    """Sanitization transliterates special chars to ASCII"""

    text = "I am sometimes not sure of what to make of Björk"
    assert sanitize_text(text) == "I sometim sure make bjork"


def test_collapse_lowers():
    """Collapsing text should cast to lowercase"""

    assert collapse_text("Testing") == "testing"


def test_collapse_removes_whitespace():
    """Collapsing text should remove whitespace"""

    assert collapse_text("Testing ") == "testing"
    assert collapse_text(" testing") == "testing"
    assert collapse_text("this is a test") == "thisisatest"
    assert (
        collapse_text(
            """
    testing"""
        )
        == "testing"
    )


def test_collapse_removes_non_alpha_chars():
    """Preparation should remove non-alpha chars"""

    assert collapse_text("Testing - 123") == "testing123"


def test_splitting_ngrams():
    """Splitting should chunk text"""

    text = "thisisjustamodernrocksong"
    chunked = create_ngrams(text, 4)

    chunked = list(chunked)

    assert chunked == ["this", "isju", "stam", "oder", "nroc", "kson", "g"]

"""
Tests fingerprinting text
"""

import mmh3

from ..fingerprinting import hash_ngram, window_ngrams


def test_ngram_hashing():
    """Hashing should hash the string in a (pos, ngram) pair"""

    value = (1, 'this')
    hashed_str = mmh3.hash('this')

    assert hash_ngram(value) == (1, hashed_str)


def test_windowing():
    """Windowing should split a sequence into sub-groups of N size"""

    ngram_hashes = [(i, i) for i in range(9)]
    windows = list(window_ngrams(ngram_hashes))

    assert len(windows) == 2
    assert windows[0] == [(i, i) for i in range(0, 4)]
    assert windows[1] == [(i, i) for i in range(4, 8)]
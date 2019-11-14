"""
Tests fingerprinting text
"""

import mmh3

from ..fingerprinting import cull_ngrams, hash_ngram, window_ngrams, winnow


def test_ngram_hashing():
    """Hashing should hash the string in a (pos, ngram) pair"""

    value = (1, "this")
    hashed_str = mmh3.hash("this")

    assert hash_ngram(value) == (1, hashed_str)


def test_cull_ngrams():
    """Culling should return only a subset of a sequence"""

    value = range(10)

    assert list(cull_ngrams(range(10), 0)) == []
    assert list(cull_ngrams(range(10), 1)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert list(cull_ngrams(range(10), 4)) == [0, 4, 8]


def test_windowing():
    """Windowing should split a sequence into sub-groups of N size"""

    ngram_hashes = [0, 1, 2, 1]
    windowed = window_ngrams(ngram_hashes, window_size=2)

    assert list(windowed) == [
        [(0, 0), (1, 1)],
        [(1, 1), (2, 2)],
        [(2, 2), (3, 1)],
    ]


def test_winnowing():
    """Winnowing selects the right-most least value"""

    windows = window_ngrams(
        [77, 74, 42, 17, 98, 50, 17, 98, 8, 88, 67, 39, 77, 74, 42, 17, 98]
    )

    winnowed_ngrams = list(winnow(windows))

    assert winnowed_ngrams == [(3, 17), (6, 17), (8, 8), (11, 39), (15, 17)]

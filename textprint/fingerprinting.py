"""
Creates text fingerprints
"""

import sys
import typing

import mmh3

from .processing import prepare_text_for_grams, split_text_into_grams


def hash_ngram(ngram: typing.Tuple[int, str]) -> typing.Tuple[int, str]:
    """Hashes given text using mmh3
    """

    hashed_text = mmh3.hash(ngram[1])

    return (ngram[0], hashed_text)


def window_ngrams(
    ngram_hashes: typing.List[typing.Tuple[int, str]], window_size: int = 4
) -> typing.List[typing.Tuple[int, str]]:
    """Creates windows of sequential ngrams of size <window_size>
    """

    for i in range(0, len(ngram_hashes) - window_size + 1, window_size):
        yield ngram_hashes[i : i + window_size]


def winnow(window: typing.List[typing.Tuple[int, str]]) -> typing.Tuple[int, str]:
    """Winnows ngram windows by selecting the minimum hash in each
    """

    least_value = (None, float("inf"))

    # select the right-most least-value hash in the window
    for value in window:
        if value[1] <= least_value[1]:
            least_value = value

    return least_value


def fingerprint_text(
    text: str,
    ngram_size: int = 5,
    ngram_retention: float = 1.0,  # 1.0 = 100% retention, 0.25 = 25%, ...
    window_size: int = 4,
) -> typing.Set[typing.Tuple[int, str]]:
    """Fingerprints given text
    """

    retainer = 1 / ngram_retention

    prepared_text: typing.List[typing.Tuple(int, str)] = prepare_text_for_grams(text)

    # split prepared text into a sequence of ngrams and their start positions
    # in the source prepared text
    ngrams = list(split_text_into_grams(prepared_text))

    # cull ngrams by scaling factor ngram_retention
    ngrams = [ngram for (i, ngram) in enumerate(ngrams) if i % retainer == 0]

    # hash each ngram, and include each position - this is the foundation of a fingerprint
    ngram_hashes = [(i, hash_ngram(ngram)) for (i, ngram) in enumerate(ngrams)]

    # window fingerprints for min hash selection
    windows = window_ngrams(ngram_hashes, window_size)

    # select fingerprints from windows
    fingerprint = set(map(winnow, windows))

    return fingerprint

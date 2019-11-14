"""
Creates text fingerprints
"""

import sys
import typing

import mmh3

from .processing import prepare_text_for_grams, split_text_into_grams


def hash_ngram(ngram: str) -> int:
    """Hashes given text using mmh3"""

    return mmh3.hash(ngram[1])


def cull_ngrams(
    ngrams: typing.List[typing.Any], modulo: int = 1
) -> typing.List[typing.Any]:
    """Culls ngrams using modulo comparison against an item's position"""

    if modulo == 0:
        return []

    return [ngram for (i, ngram) in enumerate(ngrams) if i % modulo == 0]


def window_ngrams(
    ngram_hashes: typing.List[str], window_size: int = 4
) -> typing.Generator[typing.Tuple[int, str], None, None]:
    """Creates windows of sequential ngrams of size <window_size>"""

    # inject hash position
    hashes_with_pos = [
        (pos, ngram_hash) for (pos, ngram_hash) in enumerate(ngram_hashes)
    ]

    for i in range(0, len(hashes_with_pos) - window_size + 1):
        yield hashes_with_pos[i : i + window_size]


def winnow(
    windows: typing.List[typing.Tuple[int, str]]
) -> typing.Generator[typing.Tuple[int, str], None, None]:
    """Winnows ngram windows by selecting the minimum hash in each"""

    previous_least_hash = None

    for window in windows:
        least_value = (None, float("inf"))

        # select the right-most least-value hash in the window
        for value in window:
            if value[1] <= least_value[1]:
                least_value = value

        if least_value[0] is not None and least_value != previous_least_hash:
            previous_least_hash = least_value
            yield least_value


def fingerprint_text(
    text: str,
    ngram_size: int = 5,
    cull_modulo: int = 1,  # modulo for culling ngrams (to reduce doc density)
    window_size: int = 4,
) -> typing.Set[typing.Tuple[int, str]]:
    """Fingerprints given text"""

    prepared_text = prepare_text_for_grams(text)

    # split prepared text into a sequence of ngrams and their start positions
    # in the source prepared text
    ngrams = split_text_into_grams(prepared_text)

    # cull ngrams by scaling factor ngram_retention
    ngrams = cull_ngrams(ngrams, modulo=cull_modulo)

    # hash each ngram
    ngram_hashes = map(hash_ngram, ngrams)

    # window fingerprints for min hash selection
    windows = window_ngrams(ngram_hashes, window_size)

    # select fingerprints from windows
    fingerprints = winnow(windows)

    return set(fingerprints)

"""
Text processing helpers
"""

import math
import re
import typing

from unidecode import unidecode

NON_ALPHA_RE = re.compile(r"[\W_]+", flags=re.UNICODE)


def prepare_text_for_grams(text: str) -> str:
    """Prepares text for n-gramming

    - Transliterates string to flatten special chars
    - Sanitizes, removes whitespace of all kinds (preserving control chars)

    In: This is just a modern rock song
    Out: thisisjustamodernrocksong
    """

    # transliterate text to imperialist ascii equiv
    text = unidecode(text)

    # collapse all spacing in text
    text = text.strip()
    text = text.lower()

    # remove whitespace
    text = NON_ALPHA_RE.sub("", text)

    return text


def split_text_into_grams(
    text: str, ngram_size: int = 5
) -> typing.Generator[str, None, None]:
    """Splits text into windows of <ngram_size> length

    In: thisisjustamodernrocksong, ngram size of 4
    Out: (this, isjus, tamo, ...)
    """

    gram_blocks = int(math.ceil(len(text) / ngram_size))

    for i in range(gram_blocks):
        offset = ngram_size * i
        yield text[offset : (offset + ngram_size)]

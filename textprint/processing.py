"""
Text processing helpers
"""

import math
import re
import string
import typing

from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from unidecode import unidecode

NON_ALPHA_RE = re.compile(r"[\W_]+", flags=re.UNICODE)


def sanitize_text(text: str) -> str:
    """Sanitizes text (stemming, stopword removal)

    - Strips text
    - Transliterates string to flatten special chars
    - Filters out stopworks and punctuation
    - Stems remaining words
    """

    stopwords = set(nltk_stopwords.words("english"))

    text = text.strip()

    # transliterate text to imperialist ascii equiv
    text = unidecode(text)

    sentences = sent_tokenize(text)
    tokens = map(word_tokenize, sentences)
    words = [word for sentence in tokens for word in sentence]

    # remove stopwords
    words = [word for word in words if word not in stopwords]

    # remove punctuation
    punct_table = str.maketrans("", "", string.punctuation)
    words = [word.translate(punct_table) for word in words]

    # stem remaining words
    stemmer = PorterStemmer()
    stemmed_words = map(stemmer.stem, words)

    return " ".join(stemmed_words)


def collapse_text(text: str) -> str:
    """Collapses text by removing all whitespace, and casting to lowercase

    In: This is just a modern rock song
    Out: thisisjustamodernrocksong
    """

    # collapse all spacing in text
    text = text.lower()

    # remove whitespace
    text = NON_ALPHA_RE.sub("", text)

    return text


def create_ngrams(text: str, ngram_size: int = 5) -> typing.Generator[str, None, None]:
    """Splits text into windows of <ngram_size> length

    In: thisisjustamodernrocksong, ngram size of 4
    Out: (this, isjus, tamo, ...)
    """

    text = sanitize_text(text)
    text = collapse_text(text)

    gram_blocks = int(math.ceil(len(text) / ngram_size))

    for i in range(gram_blocks):
        offset = ngram_size * i
        yield text[offset : (offset + ngram_size)]

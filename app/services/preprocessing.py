import re
import unicodedata
from collections import Counter

TOKEN_RE = re.compile(r"https?://\S+|@\w+|\w+(?:['’]\w+)?|[^\w\s]", re.UNICODE)
# Negations are deliberately absent from this set.
STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "to", "of", "in", "and", "it", "i"}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text).lower().replace("’", "'"))


def model_text(text: str) -> str:
    text = re.sub(r"https?://\S+", "http", text)
    return re.sub(r"@\w+", "@user", text)


def stem(token: str) -> str:
    # Educational suffix heuristic, deliberately not presented as linguistic lemmatization.
    for suffix in ("ing", "ed", "ly", "s"):
        if token.isalpha() and token.endswith(suffix) and len(token) > len(suffix) + 3:
            return token[: -len(suffix)]
    return token


def explore(
    text: str, remove_stopwords: bool = False, stemming: bool = False, n: int = 2
) -> dict[str, object]:
    tokens = tokenize(text)
    processed = [t for t in tokens if not remove_stopwords or t not in STOPWORDS]
    if stemming:
        processed = [stem(t) for t in processed]
    ngrams = Counter(" ".join(processed[i : i + n]) for i in range(len(processed) - n + 1))
    return {
        "original": text,
        "normalized": normalize(text),
        "tokens": tokens,
        "processed_tokens": processed,
        "processed": " ".join(processed),
        "ngrams": dict(ngrams),
        "note": (
            "Suffix stemming is educational; no lemmatizer is used. Negation words are preserved."
        ),
    }

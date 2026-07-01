import hashlib
import os
import re

_TOKEN_RE = re.compile(r"\w+")


def get_dim() -> int:
    return int(os.getenv("EMBEDDING_DIM", "256"))


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def embed_text(text: str) -> list[float]:
    dim = get_dim()
    vec = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += sign

    norm = sum(value * value for value in vec) ** 0.5
    if norm > 0:
        vec = [value / norm for value in vec]

    return vec


def get_model_name() -> str:
    return f"feature-hash-{get_dim()}"

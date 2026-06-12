from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9@._+-]+", re.IGNORECASE)


def normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def tokenize(value: str | None) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value or "")]


def token_set(value: str | None) -> set[str]:
    return set(tokenize(value))


def lexical_similarity(left: str | None, right: str | None) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def keyword_overlap_score(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    text_tokens = token_set(text)
    hits = sum(1 for keyword in keywords if keyword.lower() in text_tokens)
    return hits / len(keywords)


def most_common_sentence(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return text.strip()
    counts = Counter(sentences)
    return counts.most_common(1)[0][0]

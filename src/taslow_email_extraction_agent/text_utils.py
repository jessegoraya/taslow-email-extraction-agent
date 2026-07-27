from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9@._+-]+", re.IGNORECASE)
QUOTED_OR_FORWARDED_RE = re.compile(
    r"(?im)^\s*(?:"
    r"-{2,}\s*original message\s*-{2,}|"
    r"-{2,}\s*forwarded message\s*-{2,}|"
    r"begin forwarded message:?|"
    r"from:\s+.+@.+|"
    r"sent:\s+.+|"
    r"to:\s+.+@.+|"
    r"subject:\s+.+|"
    r"on\s+.+wrote:"
    r")\s*$"
)
FORWARDED_HANDOFF_RE = re.compile(
    r"(?:"
    r"\b(?:can|could|would|please|need\s+you\s+to|take\s+care\s+of|handle)\b"
    r".{0,120}\b(?:handle|take\s+care\s+of|follow\s+up\s+on|address|work\s+on|"
    r"respond\s+to|coordinate|complete|review|prepare|send|update)?\b"
    r".{0,80}\b(?:this|the|client|customer|original|forwarded)?\s*"
    r"(?:request|ask|item|thread|email|note|below)\b"
    r"|"
    r"\b(?:passing|sending)\s+(?:this|it)\s+along\b.{0,120}"
    r"\b(?:current\s+(?:direction|guidance|request)|work\s+from|act\s+on|"
    r"handle|review|complete|prepare|update|respond)\b"
    r"|"
    r"\b(?:current\s+(?:direction|guidance)|request\s+below)\b.{0,100}"
    r"\b(?:work\s+from|act\s+on|handle|review|complete|prepare|update|respond)\b"
    r")",
    re.IGNORECASE,
)


def normalize_text(value: str | None) -> str:
    return (value or "").strip().lower()


def newest_authored_text(value: str | None) -> str:
    """Return the newest authored email block before quoted or forwarded history."""
    newest, _quoted = split_newest_and_quoted_text(value)
    return newest


def split_newest_and_quoted_text(value: str | None) -> tuple[str, str]:
    """Split an email body into newest authored text and older quoted/forwarded context."""
    text = (value or "").strip()
    if not text:
        return "", ""
    match = QUOTED_OR_FORWARDED_RE.search(text)
    quoted_parts: list[str] = []
    if match:
        quoted_parts.append(text[match.end() :].strip())
        text = text[: match.start()].strip()
    authored_lines: list[str] = []
    quoted_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(">"):
            quoted_lines.append(re.sub(r"^\s*>\s?", "", line))
        else:
            authored_lines.append(line)
    if quoted_lines:
        quoted_parts.insert(0, "\n".join(quoted_lines).strip())
    quoted = "\n\n".join(part for part in quoted_parts if part).strip()
    return "\n".join(authored_lines).strip(), quoted


def has_forwarded_actionable_handoff(value: str | None) -> bool:
    newest, quoted = split_newest_and_quoted_text(value)
    return bool(quoted and FORWARDED_HANDOFF_RE.search(newest))


def task_context_text(body_text: str | None, task_description: str | None) -> str:
    """Return task context, including forwarded details only for explicit handoff notes."""
    description = (task_description or "").strip()
    if not has_forwarded_actionable_handoff(body_text):
        return newest_authored_text(description)
    newest, quoted = split_newest_and_quoted_text(body_text)
    parts = [description, newest, quoted]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


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

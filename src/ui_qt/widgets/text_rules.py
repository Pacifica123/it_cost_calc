from __future__ import annotations

import re

MAX_VISIBLE_WORDS = 10
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-–—][A-Za-zА-Яа-яЁё0-9]+)?")


def count_words(text: str) -> int:
    return len(_WORD_RE.findall(" ".join(text.split())))


def assert_short_text(text: str, *, field: str = "text") -> None:
    words = count_words(text)
    if words > MAX_VISIBLE_WORDS:
        raise ValueError(f"{field} must be <= {MAX_VISIBLE_WORDS} words, got {words}: {text!r}")

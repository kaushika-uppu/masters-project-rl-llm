"""Cheap, judge-free degeneracy detector for proposed proof steps (torch-free).

Catches the collapse outputs seen in practice ("/\\n,\\n,\\n,\\n,", "aaaa...", "= = = =")
without calling the judge. A step is degenerate if it is empty, or if (ignoring
whitespace) one character, one whitespace-separated token, or one short n-gram (2-4
chars) accounts for more than `dominance` of the text.
"""

from __future__ import annotations

from collections import Counter

MIN_CHARS = 4      # shorter non-whitespace text is only flagged when empty
MIN_NGRAM_CHARS = 8  # n-gram check needs this much text to be meaningful
MIN_TOKENS = 6


def is_degenerate_step(text: str | None, dominance: float = 0.5) -> bool:
    if text is None:
        return True
    compact = "".join(text.split())
    if not compact:
        return True
    n = len(compact)
    if n >= MIN_CHARS:
        if max(Counter(compact).values()) / n > dominance:
            return True
    toks = text.split()
    if len(toks) >= MIN_TOKENS:
        if max(Counter(toks).values()) / len(toks) > dominance:
            return True
    if n >= MIN_NGRAM_CHARS:
        for k in (2, 3, 4):
            grams = Counter(compact[i : i + k] for i in range(n - k + 1))
            # non-overlapping coverage estimate of the most frequent n-gram
            if max(grams.values()) * k / n > dominance:
                return True
    return False

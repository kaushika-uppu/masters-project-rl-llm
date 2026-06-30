"""State equivalence for node merging.

Tiered, conservative (bias to UNDER-merge — a false merge corrupts credit assignment;
a false split only dilutes an estimate):
  1. exact canonical-key match (cheap, always on)
  2. embedding cosine >= threshold (optional; pass an embed_fn)
  3. LLM "same proof state?" tie-break (optional; pass an llm_equiv_fn)
"""

from __future__ import annotations

from typing import Callable, Optional

from .state import canonicalize


class StateMatcher:
    def __init__(
        self,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
        llm_equiv_fn: Optional[Callable[[str, str], bool]] = None,
        cosine_threshold: float = 0.95,
    ):
        self.embed_fn = embed_fn
        self.llm_equiv_fn = llm_equiv_fn
        self.cosine_threshold = cosine_threshold

    def key(self, state_summary: str) -> str:
        return canonicalize(state_summary)

    def equivalent(self, a: str, b: str) -> bool:
        if self.key(a) == self.key(b):
            return True
        if self.embed_fn is not None:
            va, vb = self.embed_fn([a, b])
            if _cosine(va, vb) >= self.cosine_threshold:
                # optional stricter confirmation
                if self.llm_equiv_fn is None or self.llm_equiv_fn(a, b):
                    return True
        return False


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0

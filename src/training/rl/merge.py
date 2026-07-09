"""State equivalence for node merging.

Two states merge when they are the SAME proof state. Matching is:
  1. exact canonical-key match — cheap fast path / shortcut only.
  2. **semantic**: cosine similarity of state-summary embeddings >= threshold.
     This is the intended default (proof states phrased differently should merge);
     the tree caches one embedding per node and does nearest-neighbour search.
  3. optional LLM "same proof state?" tie-break on top of the embedding hit.

Conservative bias: pick the threshold so it UNDER-merges rather than over-merges — a
false merge corrupts credit assignment, a false split only dilutes an estimate.

If no `embed_fn` is supplied the matcher degrades to exact-key matching (used by the
unit-test stubs and as a deterministic fallback).
"""

from __future__ import annotations

from typing import Callable, Optional

from .state import canonicalize


class StateMatcher:
    def __init__(
        self,
        embed_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
        llm_equiv_fn: Optional[Callable[[str, str], bool]] = None,
        cosine_threshold: float = 0.9,
    ):
        self.embed_fn = embed_fn
        self.llm_equiv_fn = llm_equiv_fn
        self.cosine_threshold = cosine_threshold

    @property
    def has_embedder(self) -> bool:
        return self.embed_fn is not None

    def key(self, state_summary: str) -> str:
        return canonicalize(state_summary)

    def embed(self, state_summary: str) -> list[float]:
        """Embed one state summary (returns a single vector)."""
        return self.embed_fn([state_summary])[0]

    def confirm(self, a: str, b: str) -> bool:
        """Optional LLM tie-break after an embedding hit (True if no tie-break set)."""
        return self.llm_equiv_fn is None or self.llm_equiv_fn(a, b)

    def equivalent(self, a: str, b: str) -> bool:
        if self.key(a) == self.key(b):
            return True
        if self.embed_fn is not None:
            va, vb = self.embed_fn([a, b])
            if cosine(va, vb) >= self.cosine_threshold and self.confirm(a, b):
                return True
        return False


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = sum(x * x for x in a) ** 0.5
    db = sum(y * y for y in b) ** 0.5
    return num / (da * db) if da and db else 0.0


_cosine = cosine  # backward-compat alias

"""Proof-state representation and canonical key.

Without Lean, the "state" is the judge-maintained NL summary of (established results,
current goal). The canonical key is what node merging (merge.py) and loop detection
(reward/redundancy) compare on. Keying on STATE (not step text) gives sibling de-dup,
cross-path convergence, and loop detection from one key (design doc 4.2).
"""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def canonicalize(state_summary: str) -> str:
    """Cheap normalization for exact-match keying.

    Lowercase, strip punctuation, collapse whitespace. Conservative: this only catches
    trivial surface differences. Semantic equivalence (embedding/LLM) is layered in
    merge.py. Under-merging is the intended bias.
    """
    s = state_summary.strip().lower()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s)
    return s.strip()


def state_key(state_summary: str) -> str:
    return canonicalize(state_summary)

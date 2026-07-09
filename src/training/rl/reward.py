"""Reward components (design doc 4.1): correctness-gated novelty, path-agnostic.

- endpoint_reward: verifiable verdict-vs-label gate (the un-gameable anchor).
- step_validity_reward: per-step soundness (from the judge).
- redundancy_penalty: objective loop/state-revisit penalty (judge-free).
- novelty_bonus: rarity of a move, GATED on correctness (never reward creative nonsense).
- aggregate_step_reward: weighted combination.

The reference proof is NOT used here (matching it suppresses alternative correct paths);
it is only for offline judge calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

from .tree import Node
from .types import Problem


@dataclass
class RewardWeights:
    validity: float = 1.0
    redundancy: float = 1.0
    novelty: float = 0.3
    correct: float = 1.0
    incorrect: float = 0.0
    fail: float = 0.0


def endpoint_reward(
    problem: Problem, node: Node, w: RewardWeights = RewardWeights()
) -> float:
    """Verifiable terminal signal: did the rollout reach the correct verdict?"""
    if node.failed or node.verdict is None:
        return w.fail
    truth = "PROVED" if problem.label else "DISPROVED"
    return w.correct if node.verdict.upper() == truth else w.incorrect


def step_validity_reward(valid: bool, w: RewardWeights = RewardWeights()) -> float:
    return w.validity if valid else -w.validity


def redundancy_penalty(
    path_keys: list[str], w: RewardWeights = RewardWeights()
) -> float:
    """Penalize literal state-revisits on a path (loops). Judge-free, objective."""
    seen, dup = set(), 0
    for k in path_keys:
        if k in seen:
            dup += 1
        seen.add(k)
    return -w.redundancy * dup


def novelty_bonus(
    edge_count: int, correct: bool, w: RewardWeights = RewardWeights()
) -> float:
    """Rarity bonus for a move, GATED on the rollout being correct.

    edge_count = how many rollouts took this move from this parent (post-merge).
    Rarer (smaller count) -> larger bonus. Zero if the rollout was not correct.
    """
    if not correct or edge_count <= 0:
        return 0.0
    return w.novelty / edge_count


def aggregate_step_reward(validity: float, redundancy: float, novelty: float) -> float:
    return validity + redundancy + novelty

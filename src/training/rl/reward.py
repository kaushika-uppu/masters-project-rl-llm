"""Reward components (design doc 4.1): correctness anchor + dense, NON-gated shaping.

- endpoint_reward: verifiable verdict-vs-label gate (the un-gameable anchor).
- step_validity_reward: per-step soundness (from the judge).
- aggregate_step_reward: mean per-step validity reward over a rollout.
- redundancy_penalty: objective loop/state-revisit penalty (judge-free).
- novelty_bonus: rarity of a move, gated by `RewardWeights.novelty_gate`.
- novelty_eligible: decides whether a rollout's rare moves earn the bonus.
- proof_reward: ADDITIVE terminal reward = endpoint + validity + progress.

The reference proof is NOT used here (matching it suppresses alternative correct paths);
it is only for offline judge calibration.

Why ADDITIVE (the old form was `endpoint_reward * verifier_score`): the product zeroed
every shaping term whenever the final verdict was wrong, so on hard problems — where
almost every rollout is wrong — the tree's per-step machinery was multiplying a signal
that was almost always 0, and nothing taught *good reasoning* short of a correct verdict.
The additive form keeps correctness dominant (keep `w.correct` larger than the whole
shaping budget) while letting sound-but-incomplete reasoning earn gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

from .tree import Node
from .types import Problem, StepJudgement


@dataclass
class RewardWeights:
    # -- terminal anchor. Keep `correct` larger than the whole shaping budget below,
    #    so a correct proof still beats the most polished wrong one. --
    correct: float = 1.0
    incorrect: float = 0.0
    fail: float = 0.0
    # -- dense shaping. ADDITIVE, never gated on the final verdict --
    validity: float = 0.3          # scale of the mean per-step validity reward
    reached_verdict: float = 0.1   # small bonus for terminating with ANY verdict
    # Per-step penalty for revisiting a state (sample_builder). Kept well below `correct`
    # (1.0): at -1.0 a single revisit cancelled an entire correct verdict.
    redundancy: float = 0.2
    # Anti-guessing: a rollout whose terminal verdict arrives after FEWER than `min_steps`
    # non-terminal reasoning steps is treated like a failure (endpoint = `fail`, no
    # reached_verdict bonus, no validity shaping, not novelty-eligible). 0 disables.
    min_steps: int = 1
    # -- per-step reasoning signals (sample_builder), added directly to each step's advantage --
    # +step_validity if the judge accepted the step on the FIRST try, +revised_step_validity
    # if it was accepted only after revision. Not granted on premature rollouts.
    step_validity: float = 0.2
    revised_step_validity: float = 0.1
    # Each judge-REJECTED attempt becomes a negative training sample with advantage
    # -invalid_step (if train_on_rejected).
    invalid_step: float = 0.3
    train_on_rejected: bool = True
    # -- novelty --
    novelty: float = 0.3
    # "correct" = old behaviour (bonus only when the verdict is right)
    # "valid"   = bonus when correct OR the rollout's steps are sound (default)
    # "none"    = ungated (use with care: can reward creative nonsense)
    novelty_gate: str = "valid"
    novelty_min_validity: float = 1.0  # mean step-validity needed for the "valid" gate

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "RewardWeights":
        """Build from a (yaml) mapping; unknown keys raise so typos don't silently no-op."""
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(
                f"Unknown rl.reward keys {bad}; valid keys: {sorted(known)}"
            )
        return cls(**d)


def is_premature_verdict(
    node: Node, n_reasoning_steps: Optional[int], w: RewardWeights = RewardWeights()
) -> bool:
    """True if the rollout reached a verdict with fewer than `w.min_steps` non-terminal
    reasoning steps (a guess). `n_reasoning_steps=None` (unknown) -> never premature."""
    if n_reasoning_steps is None or w.min_steps <= 0:
        return False
    if node.failed or node.verdict is None:
        return False
    return n_reasoning_steps < w.min_steps


def endpoint_reward(
    problem: Problem, node: Node, w: RewardWeights = RewardWeights()
) -> float:
    """Verifiable terminal signal: did the rollout reach the correct verdict?"""
    if node.failed or node.verdict is None:
        return w.fail
    truth = "PROVED" if problem.label else "DISPROVED"
    return w.correct if node.verdict.upper() == truth else w.incorrect


def verifier_score(judgements: list[StepJudgement]) -> float:
    if not judgements:
        return 0.0
    valid = sum(1 for j in judgements if j.valid)
    return valid / len(judgements)


def proof_reward(
    problem: Problem,
    node: Node,
    judgements: list[StepJudgement],
    w: RewardWeights = RewardWeights(),
    n_reasoning_steps: Optional[int] = None,
) -> float:
    """ADDITIVE terminal reward for one rollout.

    = endpoint (correctness anchor) + mean step validity + progress bonus.

    Replaces the old `endpoint_reward * verifier_score` product, which collapsed to 0
    for every wrong verdict regardless of how sound the reasoning was.
    """
    if is_premature_verdict(node, n_reasoning_steps, w):
        # Guessed a verdict with too little reasoning: treated like a failure.
        return w.fail
    endpoint = endpoint_reward(problem, node, w)
    validity = aggregate_step_reward(judgements, w)
    reached = 0.0 if (node.failed or node.verdict is None) else w.reached_verdict
    return endpoint + validity + reached


def step_validity_reward(valid: bool, w: RewardWeights = RewardWeights()) -> float:
    """Per-step soundness signal: +w.validity if the judge accepted the step, else -w.validity."""
    return w.validity if valid else -w.validity


def aggregate_step_reward(
    judgements: list[StepJudgement], w: RewardWeights = RewardWeights()
) -> float:
    """Mean per-step validity reward over a rollout (in [-w.validity, +w.validity]).

    Judge-based and NOT gated on the final verdict, so a sound-but-wrong proof is still
    shaped toward sound reasoning. Empty judgement list -> 0.0.
    """
    if not judgements:
        return 0.0
    return sum(step_validity_reward(j.valid, w) for j in judgements) / len(judgements)


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
    edge_count: int, eligible: bool, w: RewardWeights = RewardWeights()
) -> float:
    """Rarity bonus for a move, earned only when `eligible` (see `novelty_eligible`).

    edge_count = how many rollouts took this move from this parent (post-merge).
    Rarer (smaller count) -> larger bonus. Zero when not eligible.
    """
    if not eligible or edge_count <= 0:
        return 0.0
    return w.novelty / edge_count


def novelty_eligible(
    success: bool,
    validity_frac: float,
    w: RewardWeights = RewardWeights(),
    premature: bool = False,
) -> bool:
    """Should this rollout's rare moves earn the novelty bonus?

    Controlled by `RewardWeights.novelty_gate`:
      "correct" -> only when the final verdict is right (old behaviour)
      "valid"   -> when correct OR the steps are sound (default; keeps exploration alive
                   on hard problems without rewarding nonsense)
      "none"    -> always
    """
    if premature:  # a guessed verdict never earns novelty, whatever the gate
        return False
    gate = w.novelty_gate
    if gate == "none":
        return True
    if gate == "correct":
        return success
    return success or validity_frac >= w.novelty_min_validity

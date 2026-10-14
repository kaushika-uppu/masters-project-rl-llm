"""Rollout engine: builds the search tree by sampling steps, judging them, retrying
failed steps in-context (self-correction), and pooling terminal rewards into MC values.

`build_tree` runs `group_size` independent rollouts from the root (the GRPO group) in
LOCK-STEP by depth: at each depth, ALL active rollouts' next steps are generated in one
batched call, and judged in one batched call. This is the big throughput win at scale —
one model forward per depth (batch = #active rollouts) instead of one per rollout per
depth. Siblings stay independent (each rollout conditions only on its own history); node
merging (tree.py) de-duplicates shared states and pools their terminal rewards.

Batching is opportunistic: if the Policy/Judge expose `*_batch` methods (the in-process
models do), they're used; otherwise the engine falls back to per-item calls (keeps the
served clients and the test stubs working).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .merge import StateMatcher
from .policy import revise_extra_user
from .degeneracy import is_degenerate_step
from .reward import RewardWeights, is_premature_verdict, proof_reward, verifier_score
from .tree import Node, ProofTree
from .types import Judge, Policy, Problem, StepJudgement


@dataclass
class RejectedStep:
    """A judge-rejected attempt: what it was generated under and why it failed."""
    parent_key: str
    history: list[str]      # applied steps before this attempt
    extra_user: str         # "" for a first attempt, the revise prompt for a retry
    step: str
    reason: str = ""


@dataclass
class Trajectory:
    path_keys: list[str]
    terminal: Node
    steps: list[str]
    success: bool = False
    reward: float = 0.0
    judgements: list[StepJudgement] = field(default_factory=list)
    verifier_score: float = 0.0
    # Extra user text each applied step was GENERATED under ("" for a plain proposal, the
    # "Your previous step was rejected..." text for a revision). Aligned with `steps`, so
    # training scores log-probs under the real generation prompt.
    step_extras: list[str] = field(default_factory=list)
    n_reasoning_steps: int = 0   # non-terminal steps taken (the verdict step excluded)
    premature: bool = False      # reached a verdict with < weights.min_steps reasoning steps
    # Per applied step: accepted by the judge on the first try (no revision)? Aligned with `steps`.
    step_first_try: list[bool] = field(default_factory=list)
    rejected: list[RejectedStep] = field(default_factory=list)
    n_proposed: int = 0          # depths at which this rollout proposed a step
    n_first_try_valid: int = 0   # ...of which the first proposal was accepted
    n_attempts: int = 0          # all generated attempts (proposals + revisions)
    n_degenerate: int = 0        # attempts flagged by is_degenerate_step


@dataclass
class _RState:
    """Internal per-rollout state during lock-step expansion."""
    node: Node
    history: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    step_extras: list[str] = field(default_factory=list)
    step_first_try: list[bool] = field(default_factory=list)
    rejected: list[RejectedStep] = field(default_factory=list)
    n_proposed: int = 0
    n_first_try_valid: int = 0
    n_attempts: int = 0
    n_degenerate: int = 0
    done: bool = False
    terminal: Optional[Node] = None
    judgements: list[StepJudgement] = field(default_factory=list)


class RolloutEngine:
    def __init__(
        self,
        policy: Policy,
        judge: Judge,
        matcher: Optional[StateMatcher] = None,
        max_depth: int = 8,
        retry_budget: int = 2,
    ):
        self.policy = policy
        self.judge = judge
        self.matcher = matcher or StateMatcher()
        self.max_depth = max_depth
        self.retry_budget = retry_budget

    # -- batched-or-fallback wrappers -------------------------------------------
    def _propose(self, problem: Problem, histories: list[list[str]]) -> list[str]:
        fn = getattr(self.policy, "propose_steps_batch", None)
        if fn is not None:
            return fn(problem, histories)
        return [self.policy.propose_steps(problem, h, k=1)[0] for h in histories]

    def _revise(self, problem: Problem, items: list[tuple]) -> list[str]:
        """items: (history, failed_step, reason)."""
        fn = getattr(self.policy, "revise_steps_batch", None)
        if fn is not None:
            return fn(problem, items)
        return [self.policy.revise_step(problem, h, fs, r) for (h, fs, r) in items]

    def _judge(self, problem: Problem, items: list[tuple]) -> list[StepJudgement]:
        """items: (history, step)."""
        fn = getattr(self.judge, "judge_steps_batch", None)
        if fn is not None:
            return fn(problem, items)
        return [self.judge.judge_step(problem, h, s) for (h, s) in items]

    # -- lock-step batched tree build -------------------------------------------
    def build_tree(
        self,
        problem: Problem,
        group_size: int = 8,
        weights: RewardWeights = RewardWeights(),
    ) -> tuple[ProofTree, list[Trajectory]]:
        tree = ProofTree(f"GOAL: prove or disprove: {problem.statement}", self.matcher)
        states = [_RState(node=tree.root, path=[tree.root.key]) for _ in range(group_size)]

        for _ in range(self.max_depth):
            active = [s for s in states if not s.done]
            if not active:
                break

            steps = self._propose(problem, [s.history for s in active])
            judged = self._judge(problem, [(s.history, st) for s, st in zip(active, steps)])
            for s, j, st in zip(active, judged, steps):
                s.judgements.append(j)
                s.n_proposed += 1
                s.n_attempts += 1
                s.n_first_try_valid += int(j.valid)
                s.n_degenerate += int(is_degenerate_step(st))
            tries = [0] * len(active)
            extras = [""] * len(active)  # extra_user each pending step was generated under
            last_fail: dict[int, Node] = {}

            # batched bounded retry: re-generate only the still-invalid rollouts
            while True:
                invalid = [i for i, s in enumerate(active) if not s.done and not judged[i].valid]
                if not invalid:
                    break
                for i in invalid:  # keep each rejected attempt as a negative sample
                    active[i].rejected.append(RejectedStep(
                        parent_key=active[i].node.key, history=list(active[i].history),
                        extra_user=extras[i], step=steps[i], reason=judged[i].reason,
                    ))
                for i in invalid:  # record each bad move as a merged dead-end leaf
                    last_fail[i] = tree.add_transition(
                        active[i].node, steps[i], f"FAILED::{steps[i]}", failed=True
                    )
                still = []
                for i in invalid:
                    if tries[i] >= self.retry_budget:  # give up this rollout
                        s, fc = active[i], last_fail[i]
                        s.path = s.path + [fc.key]
                        s.terminal = fc
                        s.done = True
                    else:
                        still.append(i)
                if not still:
                    break
                revised = self._revise(
                    problem, [(active[i].history, steps[i], judged[i].reason) for i in still]
                )
                for k, i in enumerate(still):
                    extras[i] = revise_extra_user(steps[i], judged[i].reason)
                    steps[i] = revised[k]
                    tries[i] += 1
                rej = self._judge(problem, [(active[i].history, steps[i]) for i in still])
                for k, i in enumerate(still):
                    judged[i] = rej[k]
                    active[i].judgements.append(rej[k])
                    active[i].n_attempts += 1
                    active[i].n_degenerate += int(is_degenerate_step(steps[i]))

            # apply the valid step for each rollout that didn't exhaust its retries
            for i, s in enumerate(active):
                if s.done or not judged[i].valid:
                    continue
                j, st = judged[i], steps[i]
                child = tree.add_transition(
                    s.node, st, j.state_summary, is_terminal=j.is_terminal, verdict=j.verdict
                )
                s.history = s.history + [st]
                s.steps.append(st)
                s.step_extras.append(extras[i])
                s.step_first_try.append(tries[i] == 0)
                s.path.append(child.key)
                s.node = child
                if j.is_terminal:
                    s.terminal = child
                    s.done = True

        # finalize trajectories + pool terminal rewards into MC node values
        truth = "PROVED" if problem.label else "DISPROVED"
        trajectories: list[Trajectory] = []
        for s in states:
            terminal = s.terminal if s.terminal is not None else s.node
            traj = Trajectory(s.path, terminal, s.steps)
            traj.step_extras = list(s.step_extras)
            traj.step_first_try = list(s.step_first_try)
            traj.rejected = list(s.rejected)
            traj.n_proposed, traj.n_first_try_valid = s.n_proposed, s.n_first_try_valid
            traj.n_attempts, traj.n_degenerate = s.n_attempts, s.n_degenerate
            ends_in_verdict = (
                s.terminal is not None and terminal.is_terminal and not terminal.failed
            )
            traj.n_reasoning_steps = len(s.steps) - (1 if ends_in_verdict else 0)
            traj.premature = is_premature_verdict(terminal, traj.n_reasoning_steps, weights)
            traj.judgements = list(s.judgements)
            traj.verifier_score = verifier_score(traj.judgements)
            traj.reward = proof_reward(
                problem, terminal, traj.judgements, weights, traj.n_reasoning_steps
            )
            traj.success = (
                not traj.premature
                and not terminal.failed
                and terminal.verdict is not None
                and terminal.verdict.upper() == truth
            )
            tree.record_rollout(traj.path_keys, traj.reward)
            trajectories.append(traj)
        return tree, trajectories

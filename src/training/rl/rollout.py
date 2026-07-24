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
from .reward import RewardWeights, endpoint_reward
from .tree import Node, ProofTree
from .types import Judge, Policy, Problem, StepJudgement


@dataclass
class Trajectory:
    path_keys: list[str]
    terminal: Node
    steps: list[str]
    success: bool = False
    reward: float = 0.0


@dataclass
class _RState:
    """Internal per-rollout state during lock-step expansion."""
    node: Node
    history: list[str] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    done: bool = False
    terminal: Optional[Node] = None


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
            tries = [0] * len(active)
            last_fail: dict[int, Node] = {}

            # batched bounded retry: re-generate only the still-invalid rollouts
            while True:
                invalid = [i for i, s in enumerate(active) if not s.done and not judged[i].valid]
                if not invalid:
                    break
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
                    steps[i] = revised[k]
                    tries[i] += 1
                rej = self._judge(problem, [(active[i].history, steps[i]) for i in still])
                for k, i in enumerate(still):
                    judged[i] = rej[k]

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
            traj.reward = endpoint_reward(problem, terminal, weights)
            traj.success = (
                not terminal.failed
                and terminal.verdict is not None
                and terminal.verdict.upper() == truth
            )
            tree.record_rollout(traj.path_keys, traj.reward)
            trajectories.append(traj)
        return tree, trajectories

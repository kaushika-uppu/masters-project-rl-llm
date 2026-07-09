"""Turn a rollout tree into weighted policy-gradient training samples.

For each step on each trajectory:
  advantage = MC advantage [V(child) - V(parent)]  +  novelty bonus (gated on success)
              +  redundancy penalty (if the step revisits a seen state)
  weight    = base + criticality(parent)   (concentrate updates at decision forks)

The trainer multiplies the step's log-prob by (advantage * weight). This is the
"train on the path, especially the critical nodes" objective with no value network.
Pure/stdlib — unit-testable in the sandbox with stub rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .advantage import criticality
from .policy import build_step_messages, reconstruct_continuation
from .reward import RewardWeights, novelty_bonus
from .rollout import Trajectory
from .tree import ProofTree
from .types import Problem


@dataclass
class TrainingSample:
    prompt: str  # human-readable (debug/inspection only)
    step: str  # the extracted step text
    advantage: float
    weight: float
    messages: list = field(
        default_factory=list
    )  # chat prompt the policy generated under
    continuation: str = ""  # assistant text actually produced (scored by the trainer)
    meta: dict = field(default_factory=dict)


def default_prompt(problem: Problem, prior_steps: list[str]) -> str:
    ctx = "\n".join(f"<step>{s}</step>" for s in prior_steps)
    return f"Prove or disprove the following:\n{problem.statement}\n{ctx}".rstrip()


def build_samples(
    tree: ProofTree,
    trajectories: list[Trajectory],
    problem: Problem,
    weights: RewardWeights = RewardWeights(),
    prompt_fn: Callable[[Problem, list[str]], str] = default_prompt,
    base_weight: float = 0.1,
) -> list[TrainingSample]:
    crit = criticality(tree)
    samples: list[TrainingSample] = []
    for traj in trajectories:
        seen: set[str] = {traj.path_keys[0]} if traj.path_keys else set()
        n = min(len(traj.steps), len(traj.path_keys) - 1)
        for i in range(n):
            parent = tree.nodes[traj.path_keys[i]]
            child = tree.nodes[traj.path_keys[i + 1]]
            mc_adv = child.value - parent.value
            nov = novelty_bonus(tree.edge_count(parent, child), traj.success, weights)
            redundancy = -weights.redundancy if child.key in seen else 0.0
            seen.add(child.key)
            advantage = mc_adv + nov + redundancy
            weight = base_weight + crit.get(parent.key, 0.0)
            prior = traj.steps[:i]
            samples.append(
                TrainingSample(
                    prompt=prompt_fn(problem, prior),
                    step=traj.steps[i],
                    advantage=advantage,
                    weight=weight,
                    messages=build_step_messages(problem, prior),
                    continuation=reconstruct_continuation(traj.steps[i]),
                    meta={
                        "parent": parent.key,
                        "child": child.key,
                        "mc_adv": mc_adv,
                        "novelty": nov,
                        "success": traj.success,
                    },
                )
            )
    return samples

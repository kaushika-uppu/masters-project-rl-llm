"""Turn a rollout tree into weighted policy-gradient training samples.

For each step on each trajectory:
  advantage = MC advantage [V(child) - V(parent)]  +  novelty bonus (gated, see below)
              +  redundancy penalty (if the step revisits a seen state)
              +  step validity: +step_validity (judge accepted on first try) or
                 +revised_step_validity (accepted after revision); 0 on premature rollouts
Each judge-REJECTED attempt additionally becomes a negative sample (advantage =
-invalid_step, meta rejected=True) if `weights.train_on_rejected`.
  weight    = min(base + criticality(parent) / max_criticality_in_tree, max_weight)

The MC advantage is derived from the (now additive) `proof_reward`, so sound-but-wrong
rollouts carry a non-zero value through the tree instead of being flattened to 0.

The trainer multiplies the step's log-prob by (advantage * weight). This is the
"train on the path, especially the critical nodes" objective with no value network.
Pure/stdlib — unit-testable in the sandbox with stub rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .advantage import normalized_criticality, step_weight
from .policy import build_step_messages, reconstruct_continuation
from .reward import RewardWeights, novelty_bonus, novelty_eligible
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
    max_weight: float | None = 1.0,
) -> list[TrainingSample]:
    # Criticality is normalized to [0,1] per tree, then weight = base + crit clipped to
    # max_weight (None = clip disabled) so the root fork can't dominate by ~40x.
    crit = normalized_criticality(tree)
    samples: list[TrainingSample] = []
    for traj in trajectories:
        seen: set[str] = {traj.path_keys[0]} if traj.path_keys else set()
        n = min(len(traj.steps), len(traj.path_keys) - 1)
        for i in range(n):
            parent = tree.nodes[traj.path_keys[i]]
            child = tree.nodes[traj.path_keys[i + 1]]
            mc_adv = child.value - parent.value
            # Novelty is gated by `weights.novelty_gate` (default "valid"): rare moves on a
            # correct OR fully-sound rollout earn the bonus, so exploration keeps a gradient
            # even when the final verdict is wrong.
            nov_ok = novelty_eligible(
                traj.success, traj.verifier_score, weights, premature=traj.premature
            )
            nov = novelty_bonus(tree.edge_count(parent, child), nov_ok, weights)
            redundancy = -weights.redundancy if child.key in seen else 0.0
            seen.add(child.key)
            first_try = (
                traj.step_first_try[i] if i < len(traj.step_first_try)
                else not (traj.step_extras[i] if i < len(traj.step_extras) else "")
            )
            if traj.premature:
                valid_bonus = 0.0
            else:
                valid_bonus = (
                    weights.step_validity if first_try else weights.revised_step_validity
                )
            advantage = mc_adv + nov + redundancy + valid_bonus
            weight = step_weight(base_weight, crit.get(parent.key, 0.0), max_weight)
            prior = traj.steps[:i]
            extra = traj.step_extras[i] if i < len(traj.step_extras) else ""
            samples.append(
                TrainingSample(
                    prompt=prompt_fn(problem, prior),
                    step=traj.steps[i],
                    advantage=advantage,
                    weight=weight,
                    messages=build_step_messages(problem, prior, extra_user=extra),
                    continuation=reconstruct_continuation(traj.steps[i]),
                    meta={
                        "parent": parent.key,
                        "child": child.key,
                        "mc_adv": mc_adv,
                        "novelty": nov,
                        "success": traj.success,
                        "premature": traj.premature,
                        "revised": bool(extra),
                        "first_try": first_try,
                        "step_validity": valid_bonus,
                    },
                )
            )
        if weights.train_on_rejected:
            for rj in getattr(traj, "rejected", []):
                samples.append(
                    TrainingSample(
                        prompt=prompt_fn(problem, rj.history),
                        step=rj.step,
                        advantage=-weights.invalid_step,
                        weight=step_weight(base_weight, crit.get(rj.parent_key, 0.0), max_weight),
                        messages=build_step_messages(problem, rj.history, extra_user=rj.extra_user),
                        continuation=reconstruct_continuation(rj.step),
                        meta={"parent": rj.parent_key, "rejected": True,
                              "reason": rj.reason, "revised": bool(rj.extra_user)},
                    )
                )
    return samples

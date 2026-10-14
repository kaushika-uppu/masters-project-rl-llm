"""Monte-Carlo value estimates -> per-step advantages -> critical-node weighting.

No value network: V(node) is the pooled terminal reward of rollouts through it
(tree.Node.value). Edge advantage(parent->child) = V(child) - V(parent). A node is a
critical decision point when its children's values diverge (spread/variance) — that
falls out of the values, no classifier (design doc 4.3).
"""

from __future__ import annotations

from dataclasses import dataclass

from .tree import ProofTree


@dataclass
class EdgeAdvantage:
    parent_key: str
    child_key: str
    advantage: float
    visits: int


def edge_advantages(tree: ProofTree) -> list[EdgeAdvantage]:
    out: list[EdgeAdvantage] = []
    for parent in tree.nodes.values():
        for child_key, edge in parent.children.items():
            child = tree.nodes[child_key]
            out.append(
                EdgeAdvantage(
                    parent.key, child_key, child.value - parent.value, edge.count
                )
            )
    return out


def criticality(tree: ProofTree) -> dict[str, float]:
    """Per-node decision criticality = visit-weighted spread of child values.

    Spread (max-min) captures 'this fork's choice swings the outcome'. Weighting by
    visits down-weights forks estimated from too few rollouts (noisy).
    """
    crit: dict[str, float] = {}
    for node in tree.nodes.values():
        child_vals = [tree.nodes[ck].value for ck in node.children]
        if len(child_vals) < 2:
            crit[node.key] = 0.0
            continue
        spread = max(child_vals) - min(child_vals)
        crit[node.key] = spread * (node.visits**0.5)
    return crit


def normalized_criticality(tree: ProofTree) -> dict[str, float]:
    """Criticality rescaled per tree to [0, 1] (divide by the tree's max; unchanged if the
    max is 0). Removes the sqrt(visits) * spread scale so one decision (typically the root)
    can't dominate every update by 10-40x."""
    crit = criticality(tree)
    mx = max(crit.values(), default=0.0)
    if mx <= 0:
        return crit
    return {k: v / mx for k, v in crit.items()}


def step_weight(base_weight: float, norm_crit: float, max_weight: float | None = 1.0) -> float:
    """Per-step loss weight = base + normalized criticality, clipped to max_weight
    (None = no clip)."""
    w = base_weight + norm_crit
    return w if max_weight is None else min(w, max_weight)


def most_critical(tree: ProofTree) -> str | None:
    crit = criticality(tree)
    if not crit:
        return None
    return max(crit, key=crit.get)

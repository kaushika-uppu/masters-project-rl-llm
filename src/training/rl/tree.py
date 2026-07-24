"""Proof search tree -> DAG (nodes merged by state).

A node is identified by its proof STATE (merge.py). Multiple rollouts/paths can reach
the same node; we pool their terminal rewards to get a Monte-Carlo value estimate
V(node) = mean terminal reward of rollouts passing through it (no value network needed).
Edges carry the step texts (moves) that produced each transition + how often taken
(the visit count that drives the novelty/rarity signal).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .merge import StateMatcher, cosine


@dataclass
class ChildEdge:
    child_key: str
    steps: Counter = field(default_factory=Counter)  # move text -> times proposed
    count: int = 0                                    # rollouts traversing this edge

    def add(self, step_text: str):
        self.steps[step_text] += 1
        self.count += 1


@dataclass
class Node:
    key: str
    summary: str
    depth: int
    is_terminal: bool = False
    verdict: Optional[str] = None     # "PROVED" | "DISPROVED" if terminal
    failed: bool = False              # invalid-step dead-end leaf
    visits: int = 0                   # rollouts through this node
    value_sum: float = 0.0            # sum of terminal rewards of those rollouts
    children: dict = field(default_factory=dict)  # child_key -> ChildEdge

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class ProofTree:
    def __init__(self, root_summary: str, matcher: Optional[StateMatcher] = None):
        self.matcher = matcher or StateMatcher()
        self.nodes: dict[str, Node] = {}
        self._emb: dict[str, list[float]] = {}   # node.key -> cached state embedding
        self._alias: dict[str, str] = {}         # canonical text key -> representative node.key
        # Merge tracking (must be initialized before _intern is called)
        self.merge_count = 0  # number of times states were merged
        self.state_count = 0  # total number of states added (including merged ones)
        self.root_key = self._intern(root_summary, depth=0).key

    # -- structure ---------------------------------------------------------------
    def _update_flags(self, node: Node, depth: int, kw: dict) -> Node:
        node.is_terminal = node.is_terminal or kw.get("is_terminal", False)
        node.failed = node.failed or kw.get("failed", False)
        node.verdict = node.verdict or kw.get("verdict")
        node.depth = min(node.depth, depth)
        return node

    def _intern(self, summary: str, depth: int, **kw) -> Node:
        ck = self.matcher.key(summary)
        self.state_count += 1

        # 1. exact / previously-seen text -> same node (fast path)
        nk = self._alias.get(ck)
        if nk is not None:
            self.merge_count += 1
            return self._update_flags(self.nodes[nk], depth, kw)

        # 2. semantic nearest-neighbour merge (embedding cosine >= threshold)
        vec = None
        if self.matcher.has_embedder and self.nodes:
            vec = self.matcher.embed(summary)
            best_key, best_sim = None, -1.0
            for k, ev in self._emb.items():
                sim = cosine(vec, ev)
                if sim > best_sim:
                    best_sim, best_key = sim, k
            if (best_key is not None and best_sim >= self.matcher.cosine_threshold
                    and self.matcher.confirm(summary, self.nodes[best_key].summary)):
                self._alias[ck] = best_key       # cache so identical text short-circuits
                self.merge_count += 1
                return self._update_flags(self.nodes[best_key], depth, kw)

        # 3. new node
        node = Node(key=ck, summary=summary, depth=depth, **kw)
        self.nodes[ck] = node
        self._alias[ck] = ck
        if self.matcher.has_embedder:
            self._emb[ck] = vec if vec is not None else self.matcher.embed(summary)
        return node

    @property
    def root(self) -> Node:
        return self.nodes[self.root_key]

    def add_transition(self, parent: Node, step_text: str, child_summary: str,
                       is_terminal: bool = False, verdict: Optional[str] = None,
                       failed: bool = False) -> Node:
        child = self._intern(child_summary, parent.depth + 1,
                             is_terminal=is_terminal, verdict=verdict, failed=failed)
        edge = parent.children.get(child.key)
        if edge is None:
            edge = ChildEdge(child_key=child.key)
            parent.children[child.key] = edge
        edge.add(step_text)
        return child

    # -- value estimation --------------------------------------------------------
    def record_rollout(self, path_keys: list[str], terminal_reward: float):
        """Pool a finished rollout's terminal reward into every node on its path."""
        for k in path_keys:
            n = self.nodes[k]
            n.visits += 1
            n.value_sum += terminal_reward

    # -- queries -----------------------------------------------------------------
    def child_nodes(self, node: Node) -> list[Node]:
        return [self.nodes[ck] for ck in node.children]

    def edge_count(self, parent: Node, child: Node) -> int:
        e = parent.children.get(child.key)
        return e.count if e else 0

    def size(self) -> int:
        return len(self.nodes)

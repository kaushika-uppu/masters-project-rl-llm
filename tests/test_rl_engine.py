"""End-to-end engine test with stub policy/judge (no GPU/model/dataset needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.rl.types import Problem
from src.training.rl.stubs import StubPolicy, StubJudge
from src.training.rl.rollout import RolloutEngine
from src.training.rl.reward import endpoint_reward, novelty_bonus, redundancy_penalty
from src.training.rl.advantage import criticality, most_critical, edge_advantages

PROBLEM = Problem(id="t1", statement="all fibers have the same cardinality", label=True)


def _build():
    eng = RolloutEngine(StubPolicy(), StubJudge(), max_depth=4, retry_budget=2)
    return eng.build_tree(PROBLEM, group_size=3)


def test_sibling_merge():
    tree, _ = _build()
    # move "A" proposed by 2 rollouts must collapse to ONE edge with count 2
    root = tree.root
    a_edges = [e for e in root.children.values() if "A" in e.steps]
    assert len(a_edges) == 1
    assert a_edges[0].count == 2
    # and the two A-rollouts reach a single merged child node
    assert tree.nodes[a_edges[0].child_key].visits == 2


def test_terminal_endpoint_reward():
    tree, trajs = _build()
    proved = next(n for n in tree.nodes.values() if n.verdict == "PROVED")
    disproved = next(n for n in tree.nodes.values() if n.verdict == "DISPROVED")
    assert endpoint_reward(PROBLEM, proved) == 1.0      # correct (label True)
    assert endpoint_reward(PROBLEM, disproved) == 0.0   # wrong verdict


def test_failed_leaf_recorded():
    tree, _ = _build()
    assert any(n.failed for n in tree.nodes.values()), "invalid step should leave a failed leaf"


def test_retry_recovers():
    tree, trajs = _build()
    # the B-rollout fails once then recovers to a terminal verdict (not a failed leaf)
    b_traj = next(t for t in trajs if any("B" in s for s in t.steps) or t.terminal.verdict == "DISPROVED")
    assert b_traj.terminal.is_terminal and not b_traj.terminal.failed


def test_root_is_most_critical():
    tree, _ = _build()
    # root fork (A->correct vs B->incorrect) has the largest value spread
    assert most_critical(tree) == tree.root_key
    crit = criticality(tree)
    assert crit[tree.root_key] > 0


def test_mc_values_and_advantage():
    tree, _ = _build()
    root = tree.root
    # 2/3 rollouts correct -> root value ~0.667
    assert abs(root.value - 2/3) < 1e-9
    advs = {(a.parent_key, a.child_key): a.advantage for a in edge_advantages(tree)}
    # advantage to the correct (A) child is positive; to the wrong (B) child negative
    a_child = next(ck for ck in root.children if tree.nodes[ck].value == 1.0)
    b_child = next(ck for ck in root.children if tree.nodes[ck].value == 0.0)
    assert advs[(root.key, a_child)] > 0 > advs[(root.key, b_child)]


class _NoBatchPolicy:
    """Wraps StubPolicy but exposes ONLY the single-item protocol methods (no *_batch),
    so the engine must use its per-item fallback (served clients look like this)."""
    def __init__(self): self._p = StubPolicy()
    def propose_steps(self, problem, history, k): return self._p.propose_steps(problem, history, k)
    def revise_step(self, problem, history, fs, r): return self._p.revise_step(problem, history, fs, r)


class _NoBatchJudge:
    def __init__(self): self._j = StubJudge()
    def judge_step(self, problem, history, step): return self._j.judge_step(problem, history, step)


def test_fallback_without_batch_methods():
    # engine must work when Policy/Judge lack *_batch (per-item fallback path)
    eng = RolloutEngine(_NoBatchPolicy(), _NoBatchJudge(), max_depth=4, retry_budget=2)
    tree, trajs = eng.build_tree(PROBLEM, group_size=3)
    assert most_critical(tree) == tree.root_key
    assert abs(tree.root.value - 2 / 3) < 1e-9


def test_novelty_gated_on_correctness():
    assert novelty_bonus(edge_count=1, correct=True) > novelty_bonus(edge_count=4, correct=True)
    assert novelty_bonus(edge_count=1, correct=False) == 0.0  # gated off when wrong


def test_redundancy_penalty():
    assert redundancy_penalty(["a", "b", "a"]) < 0       # revisit penalized
    assert redundancy_penalty(["a", "b", "c"]) == 0.0


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)

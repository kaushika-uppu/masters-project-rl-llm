"""End-to-end engine test with stub policy/judge (no GPU/model/dataset needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.rl.types import Problem, StepJudgement
from src.training.rl.stubs import StubPolicy, StubJudge
from src.training.rl.rollout import RolloutEngine
from src.training.rl.reward import (
    endpoint_reward,
    novelty_bonus,
    novelty_eligible,
    proof_reward,
    redundancy_penalty,
    RewardWeights,
)
from src.training.rl.advantage import criticality, most_critical, edge_advantages

PROBLEM = Problem(id="t1", statement="all fibers have the same cardinality", label=True)


# The stub scenario verdicts after <=1 reasoning step, so the legacy-shape tests disable the
# anti-guessing rule (min_steps=0); min_steps behaviour has its own tests in test_rl_fixes.py.
W0 = RewardWeights(min_steps=0)


def _build():
    eng = RolloutEngine(StubPolicy(), StubJudge(), max_depth=4, retry_budget=2)
    return eng.build_tree(PROBLEM, group_size=3, weights=W0)


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
    # Additive reward: correct branch = 1.0 (endpoint) + 0.3 (validity) + 0.1 (progress) = 1.4;
    # wrong-but-partly-sound branch = 0.0 - 0.1 + 0.1 = 0.2. Root pools to (1.4+1.4+0.2)/3 = 1.0.
    assert abs(root.value - 1.0) < 1e-9
    advs = {(a.parent_key, a.child_key): a.advantage for a in edge_advantages(tree)}
    # the correct (A) child is worth more, and carries the positive root advantage
    a_child = max(root.children, key=lambda ck: tree.nodes[ck].value)
    b_child = min(root.children, key=lambda ck: tree.nodes[ck].value)
    assert tree.nodes[a_child].value > tree.nodes[b_child].value
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
    tree, trajs = eng.build_tree(PROBLEM, group_size=3, weights=W0)
    assert most_critical(tree) == tree.root_key
    assert abs(tree.root.value - 1.0) < 1e-9  # additive rewards pooled into the root


def test_semantic_merge():
    # differently-worded states with the same meaning should merge; distinct ones shouldn't
    from src.training.rl.merge import StateMatcher
    from src.training.rl.tree import ProofTree

    def fake_embed(texts):
        out = []
        for t in texts:
            tl = t.lower()
            out.append([1.0 if "continuous" in tl else 0.0,
                        1.0 if "bounded" in tl else 0.0,
                        1.0])  # shared component so root stays distinct
        return out

    m = StateMatcher(embed_fn=fake_embed, cosine_threshold=0.9)
    tree = ProofTree("goal: prove or disprove", m)
    a = tree.add_transition(tree.root, "s1", "f is continuous on the interval")
    b = tree.add_transition(tree.root, "s2", "the function f is continuous there")
    c = tree.add_transition(tree.root, "s3", "f is bounded")
    assert a.key == b.key, "paraphrases of the same state must merge"
    assert c.key != a.key, "a genuinely different state must not merge"
    # the merged node pooled both visits' worth of structure (one node, one edge each move)
    assert tree.edge_count(tree.root, a) >= 1


def test_exact_fallback_without_embedder():
    # no embed_fn => exact canonical-key matching (deterministic)
    from src.training.rl.merge import StateMatcher
    from src.training.rl.tree import ProofTree
    tree = ProofTree("goal", StateMatcher())
    a = tree.add_transition(tree.root, "s", "State: X established.")
    b = tree.add_transition(tree.root, "s", "state   x established")  # canonicalizes equal
    assert a.key == b.key


def test_novelty_gate():
    # "valid" gate (default): rare moves earn the bonus when the rollout is correct...
    assert novelty_bonus(edge_count=1, eligible=True) > novelty_bonus(edge_count=4, eligible=True)
    assert novelty_bonus(edge_count=1, eligible=False) == 0.0
    # ...or when its steps are sound even though the final verdict is wrong
    assert novelty_eligible(success=False, validity_frac=1.0) is True
    assert novelty_eligible(success=False, validity_frac=0.0) is False
    assert novelty_eligible(success=True, validity_frac=0.0) is True
    # "correct" gate reproduces the old (correctness-only) behaviour
    w = RewardWeights(novelty_gate="correct")
    assert novelty_eligible(success=False, validity_frac=1.0, w=w) is False


class _StubNode:
    """Minimal Node stand-in: proof_reward only reads `.failed` and `.verdict`."""

    def __init__(self, verdict=None, failed=False):
        self.verdict = verdict
        self.failed = failed


def test_proof_reward_is_additive_not_gated():
    ok = [StepJudgement(valid=True) for _ in range(4)]
    bad = [StepJudgement(valid=False) for _ in range(4)]
    # wrong verdict but every step sound -> still positive (the old product gave 0)
    wrong_sound = proof_reward(PROBLEM, _StubNode(verdict="DISPROVED"), ok)
    assert wrong_sound > 0
    # a correct proof still dominates the most polished wrong one
    correct = proof_reward(PROBLEM, _StubNode(verdict="PROVED"), ok)
    assert correct > wrong_sound
    # unsound steps are penalised relative to sound ones
    wrong_unsound = proof_reward(PROBLEM, _StubNode(verdict="DISPROVED"), bad)
    assert wrong_unsound < wrong_sound
    # a failed rollout with no verdict earns no progress bonus
    failed = proof_reward(PROBLEM, _StubNode(verdict=None, failed=True), ok)
    assert failed < wrong_sound


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

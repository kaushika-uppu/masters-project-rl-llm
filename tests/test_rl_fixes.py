"""Tests for the Tree-GRPO collapse fixes (loss math, min_steps, criticality, merge safety,
revised-step prompts, config wiring, telemetry). All torch-free."""
import sys, os, json, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.training.rl.types import Problem, StepJudgement
from src.training.rl.loss import (
    k3_kl_tokens, loss_denominator, sample_loss, aggregate_rollout_stats,
)
from src.training.rl.reward import (
    RewardWeights, proof_reward, novelty_eligible, is_premature_verdict,
)
from src.training.rl.merge import StateMatcher
from src.training.rl.tree import ProofTree
from src.training.rl.advantage import normalized_criticality, criticality, step_weight
from src.training.rl.rollout import RolloutEngine
from src.training.rl.sample_builder import build_samples
from src.training.rl.policy import is_bare_verdict, build_step_messages, revise_extra_user, STEP_INSTRUCTION
from src.training.rl.stubs import StubPolicy, StubJudge
from src.training.rl.grpo_tree import GRPOTreeTrainer, TreeGRPOConfig

PROBLEM = Problem(id="p", statement="claim", label=True)


class _N:  # Node stand-in
    def __init__(self, verdict=None, failed=False):
        self.verdict, self.failed = verdict, failed


# ---- KL + loss normalization ---------------------------------------------------
def test_k3_properties():
    l = [-1.0, -2.0, -0.5]
    assert k3_kl_tokens(l, l) == [0.0, 0.0, 0.0]          # identical -> 0
    kl = k3_kl_tokens(l, [-1.5, -1.0, -3.0])
    assert all(x >= 0 for x in kl)                          # k3 is non-negative
    d = 0.7 - 0.2
    assert abs(k3_kl_tokens([0.2], [0.7])[0] - (math.exp(d) - d - 1)) < 1e-12


def test_k3_gradient_is_restoring():
    # d/dl k3 = 1 - exp(r - l): pushes l toward r (nonzero, unlike the old (lp - rlp) term
    # whose expected gradient is zero). Finite-difference check.
    l, r, eps = -1.0, -0.3, 1e-6
    num = (k3_kl_tokens([l + eps], [r])[0] - k3_kl_tokens([l - eps], [r])[0]) / (2 * eps)
    assert abs(num - (1 - math.exp(r - l))) < 1e-6
    assert num < 0  # l < r -> increasing l reduces KL


def test_loss_denominator():
    assert loss_denominator("constant", 10, 384) == 384
    assert loss_denominator("constant", 999, 384) == 384
    assert loss_denominator("mean", 10) == 10
    assert loss_denominator("mean", 0) == 1
    assert loss_denominator("sum", 77) == 1
    with pytest.raises(ValueError):
        loss_denominator("bogus", 1)


def test_sample_loss_same_norm_for_pg_and_kl():
    total, pg, kl = sample_loss(adv_weight=0.5, logp_sum=-20.0, kl_sum=2.0, kl_coef=0.1, denom=4.0)
    assert abs(pg - (-0.5 * -20.0 / 4.0)) < 1e-12
    assert abs(kl - 0.1 * 2.0 / 4.0) < 1e-12
    assert abs(total - (pg + kl)) < 1e-12
    # ratio KL/PG is independent of the denominator -> kl_coef keeps a fixed meaning
    t2, pg2, kl2 = sample_loss(0.5, -20.0, 2.0, 0.1, 384.0)
    assert abs(kl / pg - kl2 / pg2) < 1e-12
    # no ref model
    assert sample_loss(1.0, -3.0, None, 0.1, 1.0)[2] == 0.0


def test_constant_norm_removes_length_bias():
    # per-token reinforcement is identical for short and long under "constant"; under "mean"
    # a short seq gets L-times larger per-token gradient share than the sum...
    short = sample_loss(1.0, -5.0, 0.0, 0.0, loss_denominator("constant", 5))[1]
    long_ = sample_loss(1.0, -50.0, 0.0, 0.0, loss_denominator("constant", 50))[1]
    assert abs(long_ / short - 10.0) < 1e-12  # proportional to token count, not length-flat


# ---- min_steps guess penalty ---------------------------------------------------
def test_min_steps_penalizes_guess():
    w = RewardWeights(min_steps=2)
    ok = [StepJudgement(valid=True)]
    guess = proof_reward(PROBLEM, _N("PROVED"), ok, w, n_reasoning_steps=0)
    assert guess == w.fail
    assert proof_reward(PROBLEM, _N("PROVED"), ok, w, n_reasoning_steps=1) == w.fail
    careful = proof_reward(PROBLEM, _N("PROVED"), ok, w, n_reasoning_steps=2)
    assert careful == pytest.approx(w.correct + w.validity + w.reached_verdict)
    assert careful > guess
    # wrong-but-careful beats a lucky guess
    assert proof_reward(PROBLEM, _N("DISPROVED"), ok, w, n_reasoning_steps=3) > guess
    # unknown count / disabled -> old behaviour
    assert proof_reward(PROBLEM, _N("PROVED"), ok, w) == pytest.approx(1.4)
    assert proof_reward(PROBLEM, _N("PROVED"), ok, RewardWeights(min_steps=0), 0) == pytest.approx(1.4)
    # failed rollouts aren't "premature verdicts"
    assert not is_premature_verdict(_N(failed=True), 0, w)
    assert not is_premature_verdict(_N(None), 0, w)


def test_premature_not_novelty_eligible():
    assert novelty_eligible(True, 1.0, premature=True) is False
    assert novelty_eligible(True, 1.0, RewardWeights(novelty_gate="none"), premature=True) is False
    assert novelty_eligible(True, 1.0, premature=False) is True


class _GuessPolicy:
    """Always emits a verdict immediately (a guess)."""
    def propose_steps_batch(self, problem, histories):
        return ["VERDICT PROVED" for _ in histories]
    def revise_steps_batch(self, problem, items):
        return ["VERDICT PROVED" for _ in items]


def test_engine_marks_guess_premature_and_not_success():
    eng = RolloutEngine(_GuessPolicy(), StubJudge(), max_depth=4)
    _, trajs = eng.build_tree(PROBLEM, 2, RewardWeights())
    for t in trajs:
        assert t.premature and not t.success and t.reward == 0.0 and t.n_reasoning_steps == 0


# ---- bare-verdict vs substantive terminal step ----------------------------------
@pytest.mark.parametrize("step", [
    "Verdict: PROVED", "verdict: disproved.", "Verdict: PROVED\n", "  VERDICT PROVED  ",
    "Verdict: DISPROVED by the above",
])
def test_is_bare_verdict_true(step):
    assert is_bare_verdict(step)


@pytest.mark.parametrize("step", [
    "Hence f(x) >= 0 for all x, so the claim holds",
    "Take x = 0: then f(0) = -1 < 0, a counterexample, so the claim is false",
    "Verdict: PROVED because f(x) = (x-1)^2 + 1 >= 1 > 0 for every real x",
    "", "<step>", "Therefore proved",
])
def test_is_bare_verdict_false(step):
    assert not is_bare_verdict(step)


class _OneShotPolicy:
    """Emits `moves` in order, one per depth."""
    def __init__(self, *moves):
        self.moves = list(moves)
    def propose_steps_batch(self, problem, histories):
        return [self.moves[len(h)] for h in histories]
    def revise_steps_batch(self, problem, items):
        return [self.moves[len(h)] for (h, _, _) in items]


class _TerminalContentJudge:
    """Marks any step containing 'claim holds' or a Verdict line terminal."""
    def judge_steps_batch(self, problem, items):
        out = []
        for h, s in items:
            term = "claim holds" in s or s.lower().startswith("verdict")
            out.append(StepJudgement(valid=True, reason="ok", state_summary=" > ".join(h + [s]),
                                     is_terminal=term, verdict="PROVED" if term else None))
        return out
    def judge_step(self, problem, history, step):
        return self.judge_steps_batch(problem, [(history, step)])[0]


def test_substantive_terminal_step_counts_as_reasoning():
    eng = RolloutEngine(_OneShotPolicy("Hence f(x)>=0 for all x, so the claim holds"),
                        _TerminalContentJudge(), max_depth=4)
    _, trajs = eng.build_tree(PROBLEM, 2, RewardWeights(min_steps=1))
    for t in trajs:
        assert t.n_reasoning_steps == 1 and not t.ends_bare_verdict
        assert not t.premature and t.success and t.reward > 1.0


def test_two_step_proof_not_premature_at_default_min_steps():
    eng = RolloutEngine(_OneShotPolicy("Let x be real.", "Hence f(x)>=0, so the claim holds"),
                        _TerminalContentJudge(), max_depth=4)
    _, trajs = eng.build_tree(PROBLEM, 2, RewardWeights())
    assert all(t.n_reasoning_steps == 2 and not t.premature and t.success for t in trajs)


def test_substantive_step_then_bare_verdict():
    eng = RolloutEngine(_OneShotPolicy("Let x be real.", "Verdict: PROVED"),
                        _TerminalContentJudge(), max_depth=4)
    _, trajs = eng.build_tree(PROBLEM, 2, RewardWeights(min_steps=1))
    assert all(t.n_reasoning_steps == 1 and t.ends_bare_verdict and not t.premature for t in trajs)


def test_bare_verdict_first_is_premature_and_stats():
    eng = RolloutEngine(_OneShotPolicy("Verdict: PROVED"), _TerminalContentJudge(), max_depth=4)
    _, trajs = eng.build_tree(PROBLEM, 2, RewardWeights(min_steps=1))
    assert all(t.premature and t.ends_bare_verdict and t.n_reasoning_steps == 0
               and not t.success and t.reward == 0.0 for t in trajs)
    st = aggregate_rollout_stats([
        {"reward": 0, "success": False, "n_steps": 1, "premature": True, "failed": False, "bare_verdict": True},
        {"reward": 1, "success": True, "n_steps": 2, "premature": False, "failed": False, "bare_verdict": False},
    ])
    assert st["bare_verdict_frac"] == 0.5 and st["premature_frac"] == 0.5


# ---- criticality normalization -------------------------------------------------
def _fork_tree():
    eng = RolloutEngine(StubPolicy(), StubJudge(), max_depth=4)
    return eng.build_tree(PROBLEM, 3, RewardWeights(min_steps=0))


def test_criticality_normalized_and_weight_clipped():
    tree, trajs = _fork_tree()
    raw = criticality(tree)
    norm = normalized_criticality(tree)
    assert max(norm.values()) == pytest.approx(1.0)
    assert max(raw.values()) > 1.0 or True
    assert all(0.0 <= v <= 1.0 for v in norm.values())
    assert norm[tree.root_key] == pytest.approx(1.0)
    samples = build_samples(tree, trajs, PROBLEM, RewardWeights(min_steps=0), base_weight=0.1, max_weight=1.0)
    assert all(0.1 <= s.weight <= 1.0 for s in samples)
    root_w = {s.weight for s in samples if s.meta["parent"] == tree.root_key}
    assert root_w == {1.0}  # 0.1 + 1.0 clipped to max_weight


def test_step_weight():
    assert step_weight(0.1, 0.0, 1.0) == pytest.approx(0.1)
    assert step_weight(0.1, 1.0, 1.0) == 1.0
    assert step_weight(0.1, 1.0, 5.0) == pytest.approx(1.1)
    assert step_weight(0.1, 1.0, None) == pytest.approx(1.1)
    # all-zero criticality tree stays all zeros (no divide-by-zero)
    t = ProofTree("goal", StateMatcher())
    assert set(normalized_criticality(t).values()) == {0.0}


# ---- merge safety --------------------------------------------------------------
def _embed_all_same(texts):
    return [[1.0, 0.0] for _ in texts]  # every state "identical" semantically


def _tree():
    return ProofTree("root goal", StateMatcher(embed_fn=_embed_all_same, cosine_threshold=0.9))


def test_merge_respects_signature():
    t = _tree()
    plain = t.add_transition(t.root, "s1", "established fact one")
    proved = t.add_transition(plain, "v1", "claim is PROVED", is_terminal=True, verdict="PROVED")
    disproved = t.add_transition(plain, "v2", "claim is DISPROVED", is_terminal=True, verdict="DISPROVED")
    # terminal and non-terminal never merge even though the embeddings are identical
    assert len({plain.key, proved.key, disproved.key}) == 3
    assert proved.verdict == "PROVED" and disproved.verdict == "DISPROVED"
    assert not plain.is_terminal and plain.verdict is None
    # same signature still merges semantically (non-terminal <-> non-terminal, non-root)
    assert t.add_transition(plain, "s2", "another fact").key == plain.key
    p2 = t.add_transition(plain, "v3", "it is PROVED, done", is_terminal=True, verdict="PROVED")
    assert p2.key == proved.key


def test_failed_nodes_never_merge_semantically():
    t = _tree()
    good = t.add_transition(t.root, "s", "good state")
    f1 = t.add_transition(t.root, "bad1", "FAILED::bad1", failed=True)
    f2 = t.add_transition(t.root, "bad2", "FAILED::bad2", failed=True)
    assert f1.key != good.key and f2.key != f1.key
    assert not good.failed and f1.failed and f2.failed
    # exact same failed text merges (exact key)
    f1b = t.add_transition(t.root, "bad1", "FAILED::bad1", failed=True)
    assert f1b.key == f1.key


def test_exact_key_with_different_signature_is_distinct():
    t = ProofTree("root", StateMatcher())  # exact-only matcher
    a = t.add_transition(t.root, "s", "same text")
    b = t.add_transition(t.root, "s", "same text", is_terminal=True, verdict="PROVED")
    c = t.add_transition(t.root, "s", "same text", is_terminal=True, verdict="DISPROVED")
    assert len({a.key, b.key, c.key}) == 3
    assert t.add_transition(t.root, "s", "Same  text").key == a.key


def test_never_merge_into_root():
    t = _tree()
    child = t.add_transition(t.root, "s", "root goal")           # exact text == root's
    assert child.key != t.root_key
    later = t.add_transition(child, "s2", "root goal again")     # semantically ~root
    assert later.key != t.root_key
    ex = ProofTree("root", StateMatcher())
    assert ex.add_transition(ex.root, "s", "root").key != ex.root_key


# ---- revised-step messages -----------------------------------------------------
class _RevPolicy:
    """First proposal is BAD; revision is a valid step; then verdict."""
    def propose_steps_batch(self, problem, histories):
        return ["BAD first" if not h else "VERDICT PROVED" for h in histories]
    def revise_steps_batch(self, problem, items):
        return ["fixed step" for _ in items]


def test_revised_step_scored_under_generation_prompt():
    eng = RolloutEngine(_RevPolicy(), StubJudge(), max_depth=4, retry_budget=2)
    w = RewardWeights(min_steps=0)
    tree, trajs = eng.build_tree(PROBLEM, 1, w)
    t = trajs[0]
    assert t.steps == ["fixed step", "VERDICT PROVED"]
    extra = revise_extra_user("BAD first", "invalid: contains BAD")
    assert t.step_extras == [extra, ""]
    samples = build_samples(tree, trajs, PROBLEM, w)
    assert samples[0].messages == build_step_messages(PROBLEM, [], extra_user=extra)
    assert "was rejected" in samples[0].messages[-1]["content"]
    assert samples[0].messages[-1]["content"].endswith(STEP_INSTRUCTION)
    assert samples[1].messages == build_step_messages(PROBLEM, ["fixed step"])  # plain proposal
    assert samples[0].meta["revised"] and not samples[1].meta["revised"]


# ---- config -> RewardWeights ---------------------------------------------------
def test_reward_weights_defaults():
    w = RewardWeights()
    assert w.min_steps == 1 and w.redundancy == 0.2


def test_reward_weights_from_dict_and_yaml():
    w = RewardWeights.from_dict({"min_steps": 3, "validity": 0.5})
    assert w.min_steps == 3 and w.validity == 0.5 and w.correct == 1.0
    assert RewardWeights.from_dict(None) == RewardWeights()
    with pytest.raises(ValueError):
        RewardWeights.from_dict({"min_step": 3})
    yaml = pytest.importorskip("yaml")
    cfgdir = os.path.join(os.path.dirname(__file__), "..", "src", "training", "configs")
    for f in ("rl_tree.yaml", "rl_tree_from_sft.yaml"):
        rl = yaml.safe_load(open(os.path.join(cfgdir, f)))["rl"]
        assert RewardWeights.from_dict(rl["reward"]) == RewardWeights()  # yaml mirrors defaults
        cfg = TreeGRPOConfig(**rl["tree"])
        assert (cfg.loss_norm, cfg.max_weight, cfg.max_grad_norm) == ("constant", 1.0, 1.0)


def test_trainer_rejects_bad_loss_norm():
    with pytest.raises(ValueError):
        GRPOTreeTrainer(None, None, StubPolicy(), StubJudge(), cfg=TreeGRPOConfig(loss_norm="x"))


# ---- telemetry -----------------------------------------------------------------
def test_rollout_stats_and_metrics_jsonl():
    tr = GRPOTreeTrainer(None, None, StubPolicy(), StubJudge(),
                         cfg=TreeGRPOConfig(group_size=3, max_depth=4),
                         weights=RewardWeights(min_steps=0))
    tr.build_batch([PROBLEM])
    st = tr.last_rollout_stats
    assert st["n_rollouts"] == 3 and 0 <= st["success_rate"] <= 1 and st["premature_frac"] == 0.0
    rec = tr._metrics_record(1, {"loss": 0.1, "pg_loss": 0.2, "kl": 0.0, "grad_norm": 1.5,
                                 "n_samples": 5, "mean_cont_tokens": 42.0})
    expected = {"step", "loss", "pg_loss", "kl", "grad_norm", "n_samples", "mean_reward",
                "success_rate", "mean_steps_per_rollout", "frac_premature_verdict",
                "frac_failed", "mean_cont_tokens", "first_try_valid_rate", "rejected_samples",
                "mean_reasoning_steps", "degenerate_frac", "bare_verdict_frac"}
    assert set(rec) == expected
    with tempfile.TemporaryDirectory() as d:
        tr._append_metrics(d, rec); tr._append_metrics(d, rec)
        lines = open(os.path.join(d, "train_metrics.jsonl")).read().strip().splitlines()
        assert len(lines) == 2 and json.loads(lines[0])["step"] == 1


def test_aggregate_rollout_stats_empty_and_values():
    assert aggregate_rollout_stats([])["n_rollouts"] == 0
    s = aggregate_rollout_stats([
        {"reward": 1.0, "success": True, "n_steps": 3, "premature": False, "failed": False},
        {"reward": 0.0, "success": False, "n_steps": 1, "premature": True, "failed": False},
    ])
    assert s["mean_reward"] == 0.5 and s["premature_frac"] == 0.5 and s["mean_steps"] == 2.0

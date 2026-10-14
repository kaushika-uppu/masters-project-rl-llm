"""Per-step reasoning signals: step_validity, rejected-step samples, adv_whiten, degeneracy,
rollout stats. Torch-free."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.training.rl.types import Problem
from src.training.rl.reward import RewardWeights
from src.training.rl.stubs import StubPolicy, StubJudge
from src.training.rl.rollout import RolloutEngine
from src.training.rl.sample_builder import build_samples
from src.training.rl.policy import build_step_messages, revise_extra_user
from src.training.rl.loss import whiten_advantages, aggregate_rollout_stats
from src.training.rl.degeneracy import is_degenerate_step
from src.training.rl.grpo_tree import TreeGRPOConfig, GRPOTreeTrainer

P = Problem(id="p", statement="claim", label=True)


def _run(w, root_moves=("A", "A", "B"), group=3):
    eng = RolloutEngine(StubPolicy(root_moves), StubJudge(), max_depth=4, retry_budget=2)
    tree, trajs = eng.build_tree(P, group_size=group, weights=w)
    return tree, trajs, build_samples(tree, trajs, P, w)


def _plain(**kw):
    # isolate the per-step terms: no novelty/redundancy
    return RewardWeights(min_steps=0, novelty=0.0, redundancy=0.0, **kw)


def test_trajectory_records_first_try_and_rejections():
    _, trajs, _ = _run(_plain())
    b = [t for t in trajs if t.steps and t.steps[0] == "B"][0]
    assert b.step_first_try == [True, False]          # B ok first try; verdict only after revision
    assert len(b.rejected) == 1
    r = b.rejected[0]
    assert r.step == "BAD step" and r.history == ["B"] and r.extra_user == "" and "BAD" in r.reason
    a = [t for t in trajs if t.steps and t.steps[0] == "A"][0]
    assert a.step_first_try == [True, True] and a.rejected == []
    assert b.n_proposed == 2 and b.n_first_try_valid == 1


def test_first_try_vs_revised_step_validity():
    w0 = _plain(step_validity=0.0, revised_step_validity=0.0, train_on_rejected=False)
    w1 = _plain(step_validity=0.2, revised_step_validity=0.1, train_on_rejected=False)
    _, _, s0 = _run(w0)
    _, _, s1 = _run(w1)
    assert len(s0) == len(s1)
    for a, b in zip(s0, s1):
        want = 0.1 if b.meta["revised"] else 0.2
        assert b.meta["first_try"] == (not b.meta["revised"])
        assert abs((b.advantage - a.advantage) - want) < 1e-9


def test_no_step_validity_for_premature():
    # min_steps=2: stub verdicts come after <=1 reasoning step -> all premature
    w = RewardWeights(min_steps=2, novelty=0.0, redundancy=0.0, step_validity=0.5,
                      revised_step_validity=0.5, train_on_rejected=False)
    _, trajs, samples = _run(w)
    assert all(t.premature for t in trajs)
    assert all(s.meta["step_validity"] == 0.0 for s in samples)


def test_rejected_samples_messages_advantage_weight():
    w = _plain(invalid_step=0.3, train_on_rejected=True)
    tree, trajs, samples = _run(w)
    rej = [s for s in samples if s.meta.get("rejected")]
    assert len(rej) == 1
    r = rej[0]
    assert r.advantage == pytest.approx(-0.3)
    assert r.step == "BAD step" and r.continuation == "<step>BAD step</step>"
    assert r.messages == build_step_messages(P, ["B"], extra_user="")  # first-attempt prompt
    assert 0 < r.weight <= 1.0
    assert r.meta["parent"] in tree.nodes


def test_rejected_retry_uses_revise_prompt():
    # a policy whose FIRST revision is also rejected -> second rejection carries revise prompt
    class P2(StubPolicy):
        def revise_step(self, problem, history, failed_step, reason):
            return "BAD again"
    eng = RolloutEngine(P2(("B",)), StubJudge(), max_depth=4, retry_budget=2)
    w = _plain()
    tree, trajs = eng.build_tree(P, group_size=1, weights=w)
    samples = build_samples(tree, trajs, P, w)
    rej = [s for s in samples if s.meta.get("rejected")]
    assert [s.step for s in rej] == ["BAD step", "BAD again", "BAD again"][:len(rej)]
    assert len(rej) == 3
    extra = revise_extra_user("BAD step", "invalid: contains BAD")
    assert rej[1].messages == build_step_messages(P, ["B"], extra_user=extra)
    assert rej[0].messages == build_step_messages(P, ["B"], extra_user="")
    assert all(s.advantage < 0 for s in rej)


def test_train_on_rejected_toggle_off():
    _, _, samples = _run(_plain(train_on_rejected=False))
    assert not any(s.meta.get("rejected") for s in samples)
    _, _, on = _run(_plain(train_on_rejected=True))
    assert len(on) == len(samples) + 1


def test_reward_weights_new_keys_and_defaults():
    w = RewardWeights.from_dict({"step_validity": 0.4})
    assert w.step_validity == 0.4
    d = RewardWeights()
    assert (d.step_validity, d.revised_step_validity, d.invalid_step, d.train_on_rejected) == (
        0.2, 0.1, 0.3, True)


# ---- adv_whiten ---------------------------------------------------------------
def test_whiten_modes():
    advs = [1.0, 1.0, -0.3, 0.5]
    assert whiten_advantages(advs, "none") == advs
    b = whiten_advantages(advs, "batch")
    assert abs(sum(b)) < 1e-9
    s = whiten_advantages(advs, "std_only")
    assert [x > 0 for x in s] == [x > 0 for x in advs] and s[2] < 0     # signs preserved
    ratio = [x / y for x, y in zip(s, advs)]
    assert max(ratio) - min(ratio) < 1e-9 and ratio[0] > 0              # pure scaling
    assert any(x > 0 for x in b) and any(x < 0 for x in b)
    # all-negative batch: batch mode flips half positive, std_only keeps them negative
    neg = [-0.3, -0.3, -0.31]
    assert all(x < 0 for x in whiten_advantages(neg, "std_only"))
    assert any(x > 0 for x in whiten_advantages(neg, "batch"))
    assert whiten_advantages([0.5, 0.5], "std_only") == [0.5, 0.5]       # zero std -> /1
    assert whiten_advantages([], "batch") == []
    with pytest.raises(ValueError):
        whiten_advantages(advs, "bogus")


def test_trainer_rejects_bad_adv_whiten():
    with pytest.raises(ValueError):
        GRPOTreeTrainer(None, None, StubPolicy(), StubJudge(), cfg=TreeGRPOConfig(adv_whiten="x"))
    assert TreeGRPOConfig().adv_whiten == "std_only"


# ---- degeneracy ---------------------------------------------------------------
def test_degenerate_comma_spam():
    assert is_degenerate_step("/\n,\n,\n,\n,")
    assert is_degenerate_step(",\n" * 40)
    assert is_degenerate_step("")
    assert is_degenerate_step("   \n ")
    assert is_degenerate_step("x=x=x=x=x=x=")
    assert is_degenerate_step("the the the the the the the the")


def test_not_degenerate_normal_steps():
    assert not is_degenerate_step(
        "Let f: X -> Y be a surjection. Then for any y in Y the fiber f^{-1}(y) is nonempty, "
        "so we can pick x_y in it and define g(y) = x_y.")
    assert not is_degenerate_step("Verdict: PROVED")
    assert not is_degenerate_step("Since 2n = 2(k+1) = 2k + 2, n is even.")
    assert not is_degenerate_step("x = 1")


# ---- engine stats -> telemetry --------------------------------------------------
def test_aggregate_stats_new_fields():
    rows = [
        dict(reward=1, success=True, n_steps=3, premature=False, failed=False,
             n_reasoning_steps=2, n_proposed=3, n_first_try_valid=2, n_attempts=4, n_degenerate=1),
        dict(reward=0, success=False, n_steps=1, premature=False, failed=False,
             n_reasoning_steps=1, n_proposed=1, n_first_try_valid=1, n_attempts=1, n_degenerate=0),
    ]
    st = aggregate_rollout_stats(rows)
    assert st["first_try_valid_rate"] == pytest.approx(3 / 4)
    assert st["degenerate_frac"] == pytest.approx(1 / 5)
    assert st["mean_reasoning_steps"] == 2.0           # successful rollouts only
    assert aggregate_rollout_stats([])["first_try_valid_rate"] == 0.0


def test_trainer_build_batch_reports_rejected_and_stats():
    t = GRPOTreeTrainer(None, None, StubPolicy(), StubJudge(),
                        cfg=TreeGRPOConfig(group_size=3, max_depth=4),
                        weights=RewardWeights(min_steps=0))
    samples = t.build_batch([P])
    st = t.last_rollout_stats
    assert st["rejected_samples"] == sum(1 for s in samples if s.meta.get("rejected")) == 1
    assert 0 < st["first_try_valid_rate"] < 1
    rec = t._metrics_record(1, {"loss": 0.1})
    for k in ("first_try_valid_rate", "rejected_samples", "mean_reasoning_steps", "degenerate_frac"):
        assert k in rec

"""EarlyStopper rules (pure, torch-free)."""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.training.rl.early_stop import EarlyStopper, EarlyStopConfig


def healthy(step=1, **over):
    m = dict(step=step, loss=0.1, kl=0.01, grad_norm=1.0, n_samples=10, mean_reward=0.5,
             premature_frac=0.0, mean_steps=4.0, mean_cont_tokens=100.0,
             first_try_valid_rate=0.8, degenerate_frac=0.0)
    m.update(over)
    return m


def warm(es, n=5):
    for i in range(1, n + 1):
        assert es.update(healthy(i)) is None
    return n


def feed(es, start, k, **over):
    d = None
    for i in range(start, start + k):
        d = es.update(healthy(i, **over))
        if d:
            return d, i
    return None, start + k - 1


def test_healthy_run_never_stops_and_tracks_best_ok():
    es = EarlyStopper()
    for i in range(1, 30):
        assert es.update(healthy(i)) is None
    assert es.best_ok == 29 and es.last_healthy


def test_disabled():
    es = EarlyStopper(EarlyStopConfig(enabled=False))
    assert es.update(healthy(1, loss=float("nan"))) is None


@pytest.mark.parametrize("key", ["loss", "kl", "grad_norm"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_immediate_even_in_warmup(key, bad):
    es = EarlyStopper()
    d = es.update(healthy(1, **{key: bad}))
    assert d and d.hard and d.stop and key in d.reason


def test_empty_updates_streak():
    es = EarlyStopper()
    warm(es)
    assert es.update({"step": 5, "n_samples": 0}) is None
    assert es.update({"step": 5, "n_samples": 0}) is None
    d = es.update({"step": 5, "n_samples": 0})
    assert d and d.hard and "no training samples" in d.reason
    # streak resets on a real update
    es2 = EarlyStopper()
    for _ in range(2):
        es2.update({"n_samples": 0})
    es2.update(healthy(1))
    assert es2.update({"n_samples": 0}) is None


def test_warmup_suppresses_trend_checks():
    es = EarlyStopper()
    for i in range(1, 6):  # all bad but within warmup
        assert es.update(healthy(i, premature_frac=0.9, kl=2.0, grad_norm=500, degenerate_frac=0.5)) is None
        assert not es.last_healthy
    assert es.best_ok is None


def test_patience_needs_consecutive():
    es = EarlyStopper()
    n = warm(es)
    assert es.update(healthy(6, premature_frac=0.9)) is None
    assert es.update(healthy(7, premature_frac=0.9)) is None
    assert es.update(healthy(8)) is None                      # streak broken
    assert es.update(healthy(9, premature_frac=0.9)) is None
    assert es.update(healthy(10, premature_frac=0.9)) is None
    d = es.update(healthy(11, premature_frac=0.9))
    assert d and not d.hard and d.step == 11 and "premature" in d.reason


@pytest.mark.parametrize("over,needle", [
    (dict(premature_frac=0.7), "premature"),
    (dict(mean_steps=1.0), "mean_steps"),
    (dict(mean_cont_tokens=20.0), "mean_cont_tokens"),
    (dict(kl=0.9), "kl"),
    (dict(mean_reward=0.9, first_try_valid_rate=0.5), "reward"),
    (dict(grad_norm=80.0), "grad_norm"),
    (dict(degenerate_frac=0.3), "degenerate"),
])
def test_each_trend_rule_fires(over, needle):
    es = EarlyStopper()
    n = warm(es)
    d, at = feed(es, n + 1, 3, **over)
    assert d is not None and needle in d.reason and at == n + 3 and d.stop


@pytest.mark.parametrize("over", [
    dict(premature_frac=0.5),            # absolute floor boundary: not strictly greater
    dict(mean_steps=2.0),                # exactly 0.5 x baseline
    dict(mean_cont_tokens=50.0),
    dict(kl=0.5),
    dict(mean_reward=0.9, first_try_valid_rate=0.8),   # reward up but reasoning fine
    dict(mean_reward=0.5, first_try_valid_rate=0.3),   # validity down but reward flat
    dict(grad_norm=50.0),
    dict(degenerate_frac=0.1),
])
def test_rules_do_not_fire_below_threshold(over):
    es = EarlyStopper()
    n = warm(es)
    d, _ = feed(es, n + 1, 10, **over)
    assert d is None


def test_kl_none_without_ref_model_ignored():
    es = EarlyStopper()
    n = warm(es)
    d, _ = feed(es, n + 1, 5, kl=None)
    assert d is None


def test_long_name_aliases():
    es = EarlyStopper()
    base = dict(step=0, loss=0.1, n_samples=5)
    for i in range(5):
        es.update({**base, "step": i + 1, "mean_steps_per_rollout": 4.0, "frac_premature_verdict": 0.0})
    d = None
    for i in range(5, 8):
        d = es.update({**base, "step": i + 1, "mean_steps_per_rollout": 1.0, "frac_premature_verdict": 0.0})
    assert d and "mean_steps" in d.reason


def test_configurable_thresholds_and_validation():
    es = EarlyStopper(EarlyStopConfig.from_dict({"warmup": 1, "patience": 1, "max_kl": 0.1}))
    es.update(healthy(1))
    d = es.update(healthy(2, kl=0.2))
    assert d and d.step == 2
    with pytest.raises(ValueError):
        EarlyStopConfig.from_dict({"typo": 1})
    with pytest.raises(ValueError):
        EarlyStopConfig.from_dict({"mode": "halt"})


def test_warn_mode_reports_but_does_not_stop():
    es = EarlyStopper(EarlyStopConfig(mode="warn"))
    d = es.update(healthy(1, loss=float("nan")))
    assert d and not d.stop and es.decision is None
    n = warm(EarlyStopper(EarlyStopConfig(mode="warn")))
    es = EarlyStopper(EarlyStopConfig(mode="warn"))
    warm(es)
    d, _ = feed(es, 6, 3, premature_frac=0.9)
    assert d and not d.stop and es.decision is None and es.warnings


def test_best_ok_tracks_last_healthy_step():
    es = EarlyStopper()
    warm(es)
    es.update(healthy(6))
    es.update(healthy(7, premature_frac=0.9))
    assert es.best_ok == 6 and not es.last_healthy
    es.update(healthy(8))
    assert es.best_ok == 8 and es.last_healthy


def _warm_prem(es, base):
    for i in range(1, 6):
        assert es.update(healthy(i, premature_frac=base)) is None


def test_premature_relative_high_stable_baseline_does_not_fire():
    es = EarlyStopper()
    _warm_prem(es, 0.55)                       # starting policy already high
    d, _ = feed(es, 6, 10, premature_frac=0.60)  # < 0.55 + 0.15 = 0.70
    assert d is None


def test_premature_relative_rise_above_baseline_plus_delta_fires():
    es = EarlyStopper()
    _warm_prem(es, 0.55)
    assert feed(es, 6, 2, premature_frac=0.75)[0] is None   # only 2 updates (steps 6-7)
    d, at = feed(es, 8, 1, premature_frac=0.75)              # 3rd consecutive
    assert d is not None and "premature" in d.reason and at == 8


def test_premature_absolute_floor_still_applies():
    es = EarlyStopper()
    _warm_prem(es, 0.0)                        # baseline + delta = 0.15 < floor 0.5
    assert feed(es, 6, 6, premature_frac=0.4)[0] is None
    d, _ = feed(es, 12, 3, premature_frac=0.6)
    assert d is not None and "premature" in d.reason


def test_premature_delta_configurable_and_defaults():
    c = EarlyStopConfig()
    assert c.max_premature_frac == 0.5 and c.premature_delta == 0.15
    es = EarlyStopper(EarlyStopConfig.from_dict({"premature_delta": 0.5, "max_premature_frac": 0.1}))
    _warm_prem(es, 0.2)
    assert feed(es, 6, 6, premature_frac=0.6)[0] is None    # thr 0.7
    assert feed(es, 12, 3, premature_frac=0.8)[0] is not None

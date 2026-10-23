"""Early stopping on major training pathologies (pure, torch-free).

Feed `EarlyStopper.update(metrics)` once per update with the update's metrics dict
(the train_metrics.jsonl record; a no-sample update is `{"n_samples": 0}`).

Hard failures (immediate, even during warmup): NaN/inf loss, kl or grad_norm;
`max_empty_updates` consecutive updates with n_samples == 0.

Trend checks (ignored during the first `warmup` updates; fire after `patience`
consecutive bad updates): premature-verdict fraction (relative to the warmup baseline, with an absolute floor), mean_steps / mean_cont_tokens
collapse vs the warmup baseline, KL drift, reward-hacking signature (reward up while
first-try validity down), grad-norm spikes, degenerate-step fraction.

`mode="warn"` reports the same decisions but `decision.stop` is False.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional

# Accepted alternative key names (train_metrics.jsonl uses the long names).
_ALIASES = {
    "premature_frac": ("premature_frac", "frac_premature_verdict"),
    "mean_steps": ("mean_steps", "mean_steps_per_rollout"),
}


@dataclass
class EarlyStopConfig:
    enabled: bool = True
    mode: str = "stop"                 # "stop" | "warn"
    warmup: int = 5                    # updates with no trend-based stopping; also the baseline window
    patience: int = 3                  # consecutive bad updates for trend checks
    max_empty_updates: int = 3         # consecutive n_samples == 0 updates (immediate)
    max_premature_frac: float = 0.5    # absolute floor for the premature threshold
    premature_delta: float = 0.15      # fire when premature_frac > max(baseline + delta, floor)
    min_mean_steps_ratio: float = 0.5  # mean_steps < ratio * baseline
    min_cont_tokens_ratio: float = 0.5 # mean_cont_tokens < ratio * baseline
    max_kl: float = 0.5                # mean per-token k3 KL
    reward_hack_delta: float = 0.2     # mean_reward > baseline + delta ...
    first_try_drop: float = 0.15       # ... while first_try_valid_rate < baseline - drop
    max_grad_norm_spike: float = 50.0  # pre-clip grad norm
    max_degenerate_frac: float = 0.1
    max_oom_updates: int = 5           # consecutive updates with oom_skipped > 0 (immediate; 0 = off)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "EarlyStopConfig":
        d = dict(d or {})
        known = {f.name for f in fields(cls)}
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(f"Unknown rl.early_stop keys {bad}; valid keys: {sorted(known)}")
        cfg = cls(**d)
        if cfg.mode not in ("stop", "warn"):
            raise ValueError(f"rl.early_stop.mode must be 'stop' or 'warn', got {cfg.mode!r}")
        return cfg


@dataclass
class StopDecision:
    reason: str
    step: int
    hard: bool
    stop: bool   # False in warn mode


def _get(m: dict, name: str):
    for k in _ALIASES.get(name, (name,)):
        v = m.get(k)
        if v is not None:
            return v
    return None


def _num(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _finite(v) -> bool:
    return v is not None and math.isfinite(v)


class EarlyStopper:
    BASELINE_KEYS = ("premature_frac", "mean_steps", "mean_cont_tokens", "mean_reward", "first_try_valid_rate")

    def __init__(self, cfg: Optional[EarlyStopConfig] = None):
        self.cfg = cfg or EarlyStopConfig()
        self.n_updates = 0            # all updates seen (incl. empty ones)
        self.empty_streak = 0
        self.streaks: dict[str, int] = {}
        self._base_vals: dict[str, list[float]] = {k: [] for k in self.BASELINE_KEYS}
        self.baseline: dict[str, float] = {}
        self.best_ok: Optional[int] = None   # last step that passed every check
        self.last_healthy = False            # did the most recent update pass every check?
        self.decision: Optional[StopDecision] = None
        self.warnings: list[StopDecision] = []
        self.oom_streak = 0                  # consecutive updates with any OOM-skipped work

    # ---- persistence (for trainer_state.json / resume) ----
    def state_dict(self) -> dict:
        return {
            "n_updates": self.n_updates,
            "empty_streak": self.empty_streak,
            "oom_streak": self.oom_streak,
            "streaks": dict(self.streaks),
            "base_vals": {k: list(v) for k, v in self._base_vals.items()},
            "baseline": dict(self.baseline),
            "best_ok": self.best_ok,
            "last_healthy": self.last_healthy,
        }

    def load_state_dict(self, d: Optional[dict]) -> None:
        if not d:
            return
        self.n_updates = int(d.get("n_updates", 0))
        self.empty_streak = int(d.get("empty_streak", 0))
        self.oom_streak = int(d.get("oom_streak", 0))
        self.streaks = {k: int(v) for k, v in d.get("streaks", {}).items()}
        bv = d.get("base_vals", {})
        self._base_vals = {k: [float(x) for x in bv.get(k, [])] for k in self.BASELINE_KEYS}
        self.baseline = {k: float(v) for k, v in d.get("baseline", {}).items()}
        self.best_ok = d.get("best_ok")
        self.last_healthy = bool(d.get("last_healthy", False))

    # ---- baseline (mean over the warmup updates) ----
    def _accumulate_baseline(self, m: dict) -> None:
        for k in self.BASELINE_KEYS:
            v = _num(_get(m, k))
            if _finite(v):
                self._base_vals[k].append(v)
        self.baseline = {k: sum(v) / len(v) for k, v in self._base_vals.items() if v}

    def _hard(self, m: dict) -> Optional[str]:
        for k in ("loss", "kl", "grad_norm"):
            v = _num(m.get(k))
            if v is not None and not math.isfinite(v):
                return f"{k} is {v} (non-finite)"
        if m.get("n_samples") == 0:
            self.empty_streak += 1
            if self.empty_streak >= self.cfg.max_empty_updates:
                return f"no training samples for {self.empty_streak} consecutive updates"
        else:
            self.empty_streak = 0
        # Persistent OOM: scoring OOMs (oom_skipped) and rollout OOMs (rollout_oom_skipped).
        n_oom = (_num(m.get("oom_skipped")) or 0) + (_num(m.get("rollout_oom_skipped")) or 0)
        if n_oom > 0:
            self.oom_streak += 1
            if self.cfg.max_oom_updates > 0 and self.oom_streak >= self.cfg.max_oom_updates:
                return f"OOM-skipped work in {self.oom_streak} consecutive updates"
        else:
            self.oom_streak = 0
        return None

    def _trend_flags(self, m: dict) -> dict[str, str]:
        """name -> human reason, for every trend check that is bad THIS update."""
        c, b, flags = self.cfg, self.baseline, {}
        prem = _num(_get(m, "premature_frac"))
        if _finite(prem):
            thr = max(b.get("premature_frac", 0.0) + c.premature_delta, c.max_premature_frac)
            if prem > thr:
                flags["premature"] = (
                    f"premature_frac {prem:.2f} > {thr:.2f} (max(baseline "
                    f"{b.get('premature_frac', 0.0):.2f} + {c.premature_delta}, floor {c.max_premature_frac}))")
        steps = _num(_get(m, "mean_steps"))
        if _finite(steps) and b.get("mean_steps", 0) > 0 and steps < c.min_mean_steps_ratio * b["mean_steps"]:
            flags["steps_collapse"] = (
                f"mean_steps {steps:.2f} < {c.min_mean_steps_ratio} x baseline {b['mean_steps']:.2f}")
        tok = _num(_get(m, "mean_cont_tokens"))
        if _finite(tok) and b.get("mean_cont_tokens", 0) > 0 and tok < c.min_cont_tokens_ratio * b["mean_cont_tokens"]:
            flags["tokens_collapse"] = (
                f"mean_cont_tokens {tok:.1f} < {c.min_cont_tokens_ratio} x baseline {b['mean_cont_tokens']:.1f}")
        kl = _num(_get(m, "kl"))
        if _finite(kl) and kl > c.max_kl:
            flags["kl_drift"] = f"kl {kl:.3f} > {c.max_kl}"
        rew, ftv = _num(_get(m, "mean_reward")), _num(_get(m, "first_try_valid_rate"))
        if (_finite(rew) and _finite(ftv) and "mean_reward" in b and "first_try_valid_rate" in b
                and rew > b["mean_reward"] + c.reward_hack_delta
                and ftv < b["first_try_valid_rate"] - c.first_try_drop):
            flags["reward_hack"] = (
                f"mean_reward {rew:.2f} up (baseline {b['mean_reward']:.2f}) while "
                f"first_try_valid_rate {ftv:.2f} down (baseline {b['first_try_valid_rate']:.2f})")
        gn = _num(_get(m, "grad_norm"))
        if _finite(gn) and gn > c.max_grad_norm_spike:
            flags["grad_spike"] = f"grad_norm {gn:.1f} > {c.max_grad_norm_spike}"
        dg = _num(_get(m, "degenerate_frac"))
        if _finite(dg) and dg > c.max_degenerate_frac:
            flags["degenerate"] = f"degenerate_frac {dg:.2f} > {c.max_degenerate_frac}"
        return flags

    def update(self, metrics: dict) -> Optional[StopDecision]:
        """Process one update's metrics. Returns a StopDecision (reason/step) when a rule
        fires; `decision.stop` is False in warn mode. None = nothing fired."""
        if not self.cfg.enabled:
            self.last_healthy = True
            return None
        self.n_updates += 1
        step = int(metrics.get("step", self.n_updates))
        in_warmup = self.n_updates <= self.cfg.warmup
        empty = metrics.get("n_samples") == 0

        hard = self._hard(metrics)
        if hard:
            return self._fire(hard, step, True)
        if empty:  # nothing to evaluate; a skipped update is not "healthy" progress
            self.last_healthy = False
            return None

        if in_warmup:
            self._accumulate_baseline(metrics)
        flags = self._trend_flags(metrics)
        self.last_healthy = not flags
        if self.last_healthy:
            self.best_ok = step
        for name in set(self.streaks) | set(flags):
            self.streaks[name] = self.streaks.get(name, 0) + 1 if name in flags else 0
        if in_warmup:
            self.streaks = {k: 0 for k in self.streaks}
            return None
        for name, reason in flags.items():
            if self.streaks[name] >= self.cfg.patience:
                return self._fire(f"{reason} for {self.streaks[name]} consecutive updates", step, False)
        return None

    def _fire(self, reason: str, step: int, hard: bool) -> StopDecision:
        self.last_healthy = False
        d = StopDecision(reason=reason, step=step, hard=hard, stop=self.cfg.mode == "stop")
        self.warnings.append(d)
        if d.stop:
            self.decision = d
        return d

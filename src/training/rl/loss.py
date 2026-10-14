"""Pure (torch-free at import) helpers for the policy-gradient + KL loss.

All token sums (PG and KL) are divided by the SAME denominator (`loss_denominator`), so
`kl_coef` keeps a fixed meaning relative to the PG term regardless of rollout length.

loss_norm:
  "constant" (default, Dr.GRPO style): divide by a fixed constant (the policy's
             max_new_tokens). No per-sequence length bias: a long step is not penalised
             per-token more than a short one, and a short step isn't cheap to reinforce.
  "mean":    divide by the sequence's own token count (per-sequence token mean).
  "sum":     no division (legacy behaviour; biased toward short continuations).
"""

from __future__ import annotations

import math
from typing import Sequence

LOSS_NORMS = ("constant", "mean", "sum")
ADV_WHITEN_MODES = ("batch", "none", "std_only")


def whiten_advantages(advs: Sequence[float], mode: str = "std_only") -> list:
    """Batch advantage normalisation.
      "batch"    : (a - mean) / std   (old behaviour; centres, so constant terms such as
                   the rejected-step penalty / step-validity credit wash out)
      "std_only" : a / std            (default; keeps signs: rejected stays negative,
                   valid stays positive)
      "none"     : unchanged
    std is the sample std about the batch mean; a degenerate (~0) std falls back to 1."""
    if mode not in ADV_WHITEN_MODES:
        raise ValueError(f"adv_whiten must be one of {ADV_WHITEN_MODES}, got {mode!r}")
    advs = list(advs)
    if mode == "none" or not advs:
        return advs
    mean = sum(advs) / len(advs)
    var = sum((a - mean) ** 2 for a in advs) / max(len(advs) - 1, 1)
    std = var**0.5
    if std < 1e-8:
        std = 1.0
    if mode == "batch":
        return [(a - mean) / std for a in advs]
    return [a / std for a in advs]


def loss_denominator(loss_norm: str, n_tokens: int, const: float = 384.0) -> float:
    if loss_norm == "constant":
        return float(const)
    if loss_norm == "mean":
        return float(max(n_tokens, 1))
    if loss_norm == "sum":
        return 1.0
    raise ValueError(f"loss_norm must be one of {LOSS_NORMS}, got {loss_norm!r}")


def k3_kl_tokens(logp, ref_logp):
    """Per-token k3 estimator of KL(pi || ref) on tokens sampled from pi:
        kl_t = exp(r_t - l_t) - (r_t - l_t) - 1   (>= 0, unbiased, low variance)
    where l_t = log pi(token), r_t = log ref(token). Accepts torch tensors (stays
    differentiable w.r.t. logp) or plain sequences of floats (returns a list)."""
    if hasattr(logp, "exp"):  # torch tensor
        d = ref_logp - logp
        return d.exp() - d - 1
    return [math.exp(r - l) - (r - l) - 1 for l, r in zip(logp, ref_logp)]


def sample_loss(adv_weight: float, logp_sum, kl_sum, kl_coef: float, denom: float):
    """loss = [ -(adv*weight) * sum_t logp_t  +  kl_coef * sum_t kl_t ] / denom.
    Works on floats or tensors. Returns (total, pg_part, kl_part) (kl_part already
    multiplied by kl_coef)."""
    pg = -adv_weight * logp_sum / denom
    kl = kl_coef * kl_sum / denom if kl_sum is not None else 0.0
    return pg + kl, pg, kl


def aggregate_rollout_stats(rows: Sequence[dict]) -> dict:
    """Aggregate per-rollout dicts {reward, success, n_steps, premature, failed} plus optional
    {n_proposed, n_first_try_valid, n_attempts, n_degenerate, n_reasoning_steps}."""
    n = len(rows)
    if n == 0:
        return {"n_rollouts": 0, "mean_reward": 0.0, "success_rate": 0.0,
                "mean_steps": 0.0, "premature_frac": 0.0, "failed_frac": 0.0,
                "first_try_valid_rate": 0.0, "degenerate_frac": 0.0,
                "mean_reasoning_steps": 0.0}
    prop = sum(r.get("n_proposed", 0) for r in rows)
    ftv = sum(r.get("n_first_try_valid", 0) for r in rows)
    att = sum(r.get("n_attempts", 0) for r in rows)
    deg = sum(r.get("n_degenerate", 0) for r in rows)
    succ = [r for r in rows if r["success"]]
    return {
        "n_rollouts": n,
        "mean_reward": sum(r["reward"] for r in rows) / n,
        "success_rate": len(succ) / n,
        "mean_steps": sum(r["n_steps"] for r in rows) / n,
        "premature_frac": sum(1 for r in rows if r["premature"]) / n,
        "failed_frac": sum(1 for r in rows if r["failed"]) / n,
        "first_try_valid_rate": ftv / prop if prop else 0.0,
        "degenerate_frac": deg / att if att else 0.0,
        "mean_reasoning_steps": (
            sum(r.get("n_reasoning_steps", r["n_steps"]) for r in succ) / len(succ)
            if succ else 0.0
        ),
    }

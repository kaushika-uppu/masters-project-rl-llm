# Reward & Objective — Math (tree-GRPO)

Exact formulation of how the tree-RL reward, advantages, and policy-gradient update are
computed. All tunable/undecided weights are named variables; defaults are in the table at
the end. Matches `reward.py`, `loss.py`, `advantage.py`, `sample_builder.py`, `tree.py`, `grpo_tree.py`.

## Setup / notation

For a problem `x` with ground-truth label `y ∈ {PROVED, DISPROVED}`, the model generates a
**group** of `G` rollouts. A rollout `τ` is a path of states `s₀ → s₁ → … → s_T`, where `s₀`
is the root and each transition is a proof step (a "move") `aᵢ` from `s_{i-1}` to `sᵢ`. States
are merged across rollouts (semantically), so the group forms a DAG; a node `s` can lie on
many rollouts.

- `𝒯(s)` = set of rollouts whose path passes through node `s`; `visits(s) = |𝒯(s)|`.
- `ŷ(τ)` = the terminal verdict of rollout `τ` (or ⊥ if it failed / never terminated).
- `ch(s)` = child nodes of `s`; `n(s→s')` = number of rollouts that took edge `s→s'`.
- `correct(τ) = 1` if `ŷ(τ) = y`, else `0`.

## 1. Terminal reward (additive) and the anti-guessing rule

Let `k(τ)` = number of **reasoning steps** in `τ`: all steps, excluding a terminal step only if it is a *bare verdict line* (`Verdict: PROVED`, `policy.is_bare_verdict`). A terminal step with real content (the concluding argument) counts.
A rollout is **premature** if it reached a verdict (not failed) with `k(τ) < min_steps`.

```
                ⎧ r_fail                                               if τ failed, ŷ = ⊥, or τ premature
R(τ)  =         ⎨
                ⎩ endpoint(τ) + w_val · mean_i(±1 by step validity) + r_reached
                  endpoint = r_correct if ŷ = y else r_incorrect
```

Premature rollouts get **no** endpoint credit, no validity shaping, no `reached_verdict`
bonus, are never `success`, and are never novelty-eligible. Without this, a 1-step 50/50 guess
(≈ 0.5 + 0.3 + 0.1) scored as well as a careful proof. `min_steps = 0` disables the rule; the default `min_steps = 1` only catches a verdict with zero reasoning.
The correct anchor stays larger than the whole shaping budget (`r_correct > w_val + r_reached`).

## 2. Monte-Carlo node value (no value network)

```
V(s)  =  (1 / visits(s)) · Σ_{τ ∈ 𝒯(s)} R(τ)        =  value_sum(s) / visits(s)
```

### Node merging (tree.py)
Nodes merge only with an identical signature `(failed, is_terminal, verdict)`. The signature
is part of the identity key, so exact-key hits are checked too. Failed dead-ends merge on exact
key only; nothing merges into the root (a state equal/similar to the root is a distinct node).
Semantic nearest-neighbour (cosine ≥ threshold) searches only same-signature, non-root,
non-failed nodes.

## 3. Per-step reward terms (edge `sᵢ → s_{i+1}` on rollout τ)

```
A_mc(i)  = V(s_{i+1}) − V(sᵢ)
b_nov(i) = elig(τ) · w_nov / n(sᵢ → s_{i+1})     elig = 0 if premature; else per novelty_gate
                                                  ("valid": correct OR mean validity ≥ novelty_min_validity)
p_red(i) = − w_red · 1[ s_{i+1} ∈ {s₀,…,sᵢ} ]
v(i)     = ⎧ 0                  if τ premature
           ⎨ w_sv                 if step i was accepted by the judge on the FIRST try
           ⎩ w_rsv                if step i was accepted only after revision
Ã(i)     = A_mc(i) + b_nov(i) + p_red(i) + v(i)
```
`v(i)` is the direct per-step reasoning signal: the judge's verdict on THAT step (the mean over
the rollout still also enters `R(τ)` and hence `V`). Correctness stays dominant because `A_mc`
spans up to `r_correct + shaping` ≈ 1.4 while a whole 5-step valid chain earns at most 5·0.2.

**Rejected attempts (negative samples).** Every judge-rejected attempt on τ (first try or a
retry) becomes its own sample, generated under the exact prompt it was produced with
(`build_step_messages(problem, history, extra_user)`; `extra_user` empty for a first attempt,
the rejection text for a retry):
```
Ã_rej = − w_inv        W = min( w_base + ĉ(parent), w_max )        (disabled by train_on_rejected=false)
```

## 4. Criticality weight (bounded)

```
c(s)   = ( max_{s'∈ch(s)} V(s') − min_{s'∈ch(s)} V(s') ) · sqrt(visits(s))     (0 if |ch(s)| < 2)
ĉ(s)   = c(s) / max_{s''∈tree} c(s'')          (per tree; unchanged if max = 0)   ∈ [0,1]
W(i)   = min( w_base + ĉ(sᵢ),  w_max )
```
Previously `W = w_base + c` was unbounded (≈ 4 at a group-8 root vs 0.1 elsewhere), so the
first-step decision dominated every update.

## 5. Advantage normalisation (per update batch 𝓑, `tree.adv_whiten`)

```
"std_only" (default):  Â(i) = Ã(i) / σ              signs preserved (rejected < 0, valid > 0)
"batch"    (old)    :  Â(i) = ( Ã(i) − μ ) / σ      centring washes out the constant terms
"none"              :  Â(i) = Ã(i)
```
(μ, σ = mean / sample std of Ã over the batch; σ ≈ 0 falls back to 1.)

## 6. Policy-gradient + KL objective

Token log-probs `l_t = log π_θ(token_t | cᵢ, token_{<t})`, reference `r_t` (frozen SFT policy,
same token ids, no grad). Per-token **k3** KL estimator on tokens sampled from π_θ:
```
kl_t = exp(r_t − l_t) − (r_t − l_t) − 1          (≥ 0;  ∂kl_t/∂l_t = 1 − exp(r_t − l_t), restoring)
```
(The previous term `(Σl − Σr)/L` has gradient `∇Σl / L`, whose on-policy expectation is zero —
it did not pull toward the reference.)

Both token sums share ONE normalizer `N` (`loss_norm`):
```
ℓ(i) = [ − Â(i)·W(i)·Σ_t l_t  +  β · Σ_t kl_t ] / N
N = loss_norm_const (=384, Dr.GRPO)   if loss_norm = "constant"   (default)
N = L_i (sequence token count)        if loss_norm = "mean"
N = 1                                 if loss_norm = "sum"        (legacy; biased to short steps)
L(θ) = (1/|𝓑|) Σ ℓ(i)       g ← clip_by_global_norm(∇L, max_grad_norm)       θ ← AdamW step
```
KL is only applied if a reference model is loaded. Logged per update: loss, pg_loss, mean
per-token KL, pre-clip grad norm, mean continuation tokens.

## 7. Weights / knobs

Set under `rl.reward:` (RewardWeights) and `rl.tree:` (TreeGRPOConfig) in the yaml; effective
values are printed at startup.

| Symbol | Meaning | Where | Default |
|---|---|---|---|
| `r_correct` | correct verdict | `reward.correct` | 1.0 |
| `r_incorrect` | wrong verdict | `reward.incorrect` | 0.0 |
| `r_fail` | failed / no verdict / premature | `reward.fail` | 0.0 |
| `w_val` | step-validity scale (mean in [−w, +w]) | `reward.validity` | 0.3 |
| `r_reached` | bonus for any non-premature verdict | `reward.reached_verdict` | 0.1 |
| `w_red` | revisit penalty | `reward.redundancy` | 0.2 (was 1.0) |
| `min_steps` | reasoning steps required before a verdict counts | `reward.min_steps` | 1 |
| `w_sv` | per-step credit, accepted first try | `reward.step_validity` | 0.2 |
| `w_rsv` | per-step credit, accepted after revision | `reward.revised_step_validity` | 0.1 |
| `w_inv` | advantage of a rejected attempt is `−w_inv` | `reward.invalid_step` | 0.3 |
| — | train on rejected attempts | `reward.train_on_rejected` | true |
| — | advantage normalisation | `tree.adv_whiten` | "std_only" |
| `w_nov` | novelty scale | `reward.novelty` | 0.3 |
| — | novelty gate / min validity | `reward.novelty_gate`, `novelty_min_validity` | "valid", 1.0 |
| `w_base` | floor on per-step weight | `tree.base_weight` | 0.1 |
| `w_max` | cap on per-step weight (crit normalized to [0,1]) | `tree.max_weight` | 1.0 |
| `β` | KL coefficient | `tree.kl_coef` | 0.05 |
| — | loss normalization / constant | `tree.loss_norm`, `loss_norm_const` | "constant", 384 |
| — | grad-norm clip | `tree.max_grad_norm` | 1.0 |
| `η` | learning rate | `tree.lr` | 1e-6 |
| `G` | group size | `tree.group_size` | 8 |

**Early stopping** (`rl.early_stop`, `early_stop.py`, enabled by default; `mode: stop|warn`).
Immediate: non-finite loss/kl/grad_norm; `max_empty_updates`=3 zero-sample updates in a row.
After `warmup`=5 updates (baseline = mean over warmup), for `patience`=3 consecutive updates:
premature_frac > max(warmup-baseline + `premature_delta`=0.15, `max_premature_frac`=0.5 floor); mean_steps < 0.5×baseline; mean_cont_tokens < 0.5×baseline; kl > 0.5;
reward > baseline+0.2 while first_try_valid_rate < baseline−0.15; pre-clip grad_norm > 50;
degenerate_frac > 0.1. On stop: reason printed + written to train_metrics.jsonl
(`{"early_stop": true, "reason", "step"}`), checkpoint `early-stop-<step>`; `last-healthy` is
refreshed whenever `save_every` fires on a healthy update. A non-finite grad norm also skips
the optimizer step.

**Telemetry.** `<output_dir>/train_metrics.jsonl`, one line per update: step, loss, pg_loss, kl,
grad_norm, n_samples, mean_reward, success_rate, mean_steps_per_rollout,
frac_premature_verdict, frac_failed, bare_verdict_frac (terminal step is a bare "Verdict:" line), mean_cont_tokens, first_try_valid_rate (first-proposal acceptance), rejected_samples (count), mean_reasoning_steps (successful rollouts), degenerate_frac (empty/repetitive proposed steps). Watch `mean_steps_per_rollout` falling
toward 1, `frac_premature_verdict` rising, `mean_cont_tokens` collapsing, or `kl` blowing up.

**Tuning notes.** Revised steps are scored under the rejection prompt they were generated with.
With `loss_norm=constant`, `kl_coef` is relative to a 384-token budget, so per-token KL pressure
is the same for every sequence length. `r_incorrect = r_fail = 0` means all-fail groups have
zero advantage; consider partial credit if cold-start is sparse.

## Evaluating the judge

Two steps. The held-out eval file `data/deeptheorem_eval.jsonl` has no reference proofs, so first attach them
(this writes a new file and never modifies the eval file; needs the `datasets` library / HF access):

```
python scripts/build_judge_eval_set.py   # -> data/deeptheorem_judge_eval.jsonl (adds "reference", "reference_check")
python scripts/eval_judge.py             # defaults to data/deeptheorem_judge_eval.jsonl
```

The build step prints total/written/skipped (by reason)/statement-mismatch counts and exits non-zero if nothing was
written. References: original -> `proof`, entailing/pos -> `pos["response"]`, contradictory/neg -> `neg["response"]`;
variant comes from `variant_type` or the id suffix, never from the label. `eval_judge.py` exits with a clear message
if the problems file has no references. The sbatch builds the file automatically if it is missing.

`python scripts/eval_judge.py` (or `sbatch sh_scripts/templates/eval_judge.sbatch`) scores the step judge
on a labeled set built from reference proofs of a held-out problem file: gold steps (should be accepted),
a gold terminal verdict, and synthetic bad steps of several kinds. It writes `items.jsonl`, `metrics.json`,
`report.txt`, `comparison.txt` under `results/judge_eval/`. Hand-labeled real rollout steps can be scored with
`--labeled-file` (`{statement, label?, history, step, gold_valid, kind?}` per line). Use `--stub` for a no-GPU smoke test.

How to read the metrics ("invalid" is the positive class):
- **recall (bad-step catch rate)**: fraction of bad steps the judge rejects. This is the key number for reward
  reliability; a low value means invalid steps earn reward.
- **precision**: of the steps the judge rejects, how many were really bad.
- **false-reject rate** (gold steps judged invalid): over-strictness; high values starve the policy of reward.
- **per-kind accept rate** (fraction judged valid, with Wilson 95% CI): want high for `gold`, low for every bad kind.
  High accept on `premature_verdict` (bare guess), `wrong_verdict` (opposite verdict after the full proof) or
  `degenerate` (junk) means the judge rewards exactly the failure modes seen in training (verdict-only collapse,
  repetitive junk). `negated`/`numeric_perturb` can occasionally still be valid, so treat them as softer signals.
- **parse_error_rate**: unparseable judge output (counted as invalid); high values mean the JSON format is unreliable.
- **terminal**: on the gold terminal verdict item, how often the judge sets `is_terminal` and returns the correct verdict.

Evaluate on a held-out file, not the RL training problems (the CLI warns if it looks like the training file).

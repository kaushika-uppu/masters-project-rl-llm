"""Tree-GRPO trainer: per-step advantages from the rollout tree (no value network).

Flow per train step:
  problems -> RolloutEngine builds a tree per problem (step-by-step gen + judge + retry)
           -> Monte-Carlo node values (pooled terminal rewards)
           -> build_samples: per-step (advantage = MC adv + novelty + redundancy,
              weight = base + criticality)
           -> policy-gradient update: loss = [-(adv*weight)*sum_t logp_t + kl*sum_t k3_t] / norm
              (k3 = per-token KL(model||ref) estimator; `norm` per cfg.loss_norm, see loss.py)

Generation is IN-PROCESS (no servers): the policy is the model being trained
(on-policy), the judge is a second model loaded in-process. `build_batch` is pure
(sandbox-testable); `_update`/`train` need torch + GPU (cluster).

Parallelism: `num_workers` builds multiple problems' trees concurrently. Within one
problem, rollouts stay serial (they share one mutable tree). NOTE: a single in-process
model is not thread-safe for concurrent generate — use num_workers>1 only with a
served/replicated backend or one model per worker; otherwise leave it at 1 and rely on
batched sibling generation + (for the baseline) vLLM.
"""

from __future__ import annotations

import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Optional

from .train_utils import (
    OPTIMIZER_FILE, REF_ADAPTER_NAME, TRAINER_STATE_FILE, TrainerState, continuation_span,
    cuda_indices, decode_py_random_state, encode_py_random_state, first_adapter_name,
    format_peak_mem, load_trainer_state, logits_to_keep_count, oom_exception_types,
    oom_skip_step, parse_ref_mode, publish_copy, resolve_ref_mode, resume_plan,
    save_trainer_state, trainable_param_names, truncate_metrics_after,
)

from .inproc import TransformersJudge, TransformersPolicy
from .merge import StateMatcher
from .early_stop import EarlyStopConfig, EarlyStopper
from .loss import (
    ADV_WHITEN_MODES, LOSS_NORMS, aggregate_rollout_stats, k3_kl_tokens, loss_denominator,
    sample_loss, whiten_advantages,
)
from .problems import load_problems_jsonl
from .reward import RewardWeights
from .rollout import RolloutEngine
from .sample_builder import TrainingSample, build_samples
from .types import Judge, Policy, Problem

# NOTE: torch and transformers are imported lazily inside the methods that need them
# (_seq_logprob / _update / _load_judge_model). This is deliberate: it keeps this module
# importable without a GPU/torch install so the pure tree-building logic (build_batch,
# _tree_samples) stays unit-testable in the sandbox (see test_trainer_build_batch_no_torch).


_FWD_MODE_CACHE: dict = {}  # model class -> (mode, kwarg), see GRPOTreeTrainer._forward_mode


def dropout_modules_to_eval(model, is_dropout=None) -> list:
    """Modules under `model.modules()` that must be put in eval() after model.train() so
    dropout stays off (torch-free: matches on class name unless `is_dropout` is given)."""
    if is_dropout is None:
        def is_dropout(m):  # nn.Dropout, Dropout1d/2d/3d, AlphaDropout, ...
            return "dropout" in type(m).__name__.lower()
    return [m for m in model.modules() if is_dropout(m)]


def is_oom_error(e: BaseException) -> bool:
    """torch.OutOfMemoryError / torch.cuda.OutOfMemoryError (same class on torch>=2.5),
    detected by class name so this stays torch-free."""
    return any(c.__name__ == "OutOfMemoryError" for c in type(e).__mro__)


def _free_cuda_cache() -> None:
    """gc + empty_cache; no-op without torch/CUDA. Call OUTSIDE an `except` block so the
    exception's traceback no longer pins the tensors that caused the OOM."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _common_prefix_len(a: list, b: list) -> int:
    """Length of the shared leading token-id prefix (robust prompt/continuation split)."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@dataclass
class TreeGRPOConfig:
    group_size: int = 8
    max_depth: int = 8
    retry_budget: int = 2
    kl_coef: float = 0.05
    lr: float = 1e-6
    epochs: int = 1
    batch_problems: int = 4
    base_weight: float = 0.1
    # Criticality is normalized to [0,1] per tree; weight = min(base_weight + crit, max_weight).
    max_weight: float = 1.0
    # Length normalization of the PG and KL token sums (see loss.py): "constant" divides by
    # `loss_norm_const` (Dr.GRPO), "mean" by the sequence's token count, "sum" = legacy.
    loss_norm: str = "constant"
    loss_norm_const: float = 384.0  # = policy max_new_tokens
    max_grad_norm: float = 1.0      # global grad-norm clip before opt.step() (<=0 disables)
    # Batch advantage normalisation: "std_only" (a/std, keeps signs so rejected-step negatives
    # and valid-step positives survive), "batch" ((a-mean)/std, old behaviour), "none".
    adv_whiten: str = "std_only"
    num_workers: int = 1  # parallel trees across problems (see module note)
    # Skip scoring any sample whose rendered (prompt+continuation) token length exceeds
    # this, to cap peak per-sequence memory in _update. 0 = disabled (score everything).
    # A single very long tree rollout is what tips a marginal GPU into OOM mid-run.
    max_score_tokens: int = 0
    # Toggle activation checkpointing on the policy during the update pass. Trades ~30%
    # compute for a large activation-memory saving across the decoder layers. Rollouts
    # (generation) keep use_cache=True and are unaffected because we only toggle in _update.
    grad_checkpointing: bool = True
    # If MORE than this fraction of a batch's samples OOM during scoring, skip opt.step() for
    # that batch (grads zeroed) instead of stepping on a heavily truncated gradient.
    max_oom_skip_frac: float = 0.25


class GRPOTreeTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        policy: Policy,
        judge: Judge,
        ref_model=None,
        cfg: TreeGRPOConfig = TreeGRPOConfig(),
        matcher: Optional[StateMatcher] = None,
        weights: Optional[RewardWeights] = None,
        early_stop: Optional[EarlyStopConfig] = None,
        ref_adapter: Optional[str] = None,
        mem_devices: Optional[list] = None,
    ):
        """KL reference: either `ref_model` (separate frozen model, any device) OR
        `ref_adapter` (name of a frozen PEFT adapter already loaded on `model`; ref log-probs
        are computed by temporarily switching `model` to that adapter). Not both."""
        if ref_model is not None and ref_adapter is not None:
            raise ValueError("pass either ref_model or ref_adapter, not both")
        if cfg.loss_norm not in LOSS_NORMS:
            raise ValueError(f"loss_norm must be one of {LOSS_NORMS}, got {cfg.loss_norm!r}")
        if cfg.adv_whiten not in ADV_WHITEN_MODES:
            raise ValueError(f"adv_whiten must be one of {ADV_WHITEN_MODES}, got {cfg.adv_whiten!r}")
        self.stopper = EarlyStopper(early_stop if early_stop is not None else EarlyStopConfig())
        self.early_stop_info: Optional[dict] = None
        self.last_healthy_step: Optional[int] = None
        self.last_healthy_path: Optional[str] = None
        weights = weights if weights is not None else RewardWeights()
        self._stats_lock = threading.Lock()
        self._rollout_rows: list[dict] = []
        self.last_rollout_stats: dict = aggregate_rollout_stats([])
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model
        self.ref_adapter = ref_adapter
        self.mem_devices = list(mem_devices or [])
        self.cfg = cfg
        self.weights = weights
        self.engine = RolloutEngine(
            policy, judge, matcher, cfg.max_depth, cfg.retry_budget
        )
        # Resume / progress cursor (saved in trainer_state.json).
        self._step = 0
        self._problem_index = 0
        self._epoch = 0
        self._n_problems = 0
        self._rollout_oom = 0
        self._policy_adapter = "default"
        self._trainable_names: Optional[set] = None
        if model is not None and hasattr(model, "peft_config"):
            self._policy_adapter = first_adapter_name(getattr(model, "active_adapter", None))
        if ref_adapter is not None:
            if model is None:
                raise ValueError("ref_adapter requires a model")
            if self._policy_adapter == ref_adapter:
                raise ValueError(f"policy adapter and ref adapter share the name {ref_adapter!r}")
            # Snapshot the policy's trainable params once (ref adapter excluded by name). Every
            # time we switch adapters we re-assert exactly this set, because PEFT's set_adapter
            # flips requires_grad for adapter params.
            self._trainable_names = set(
                trainable_param_names(
                    [(n, p.requires_grad) for n, p in model.named_parameters()], ref_adapter
                )
            )
            self._enforce_trainable()

    @property
    def _kl_enabled(self) -> bool:
        return self.ref_model is not None or self.ref_adapter is not None

    # -- PEFT adapter switching for the shared-base KL reference -----------------
    def _enforce_trainable(self) -> None:
        """Only the policy adapter's snapshot params require grad; ref params never do."""
        if self._trainable_names is None:
            return
        for n, p in self.model.named_parameters():
            want = n in self._trainable_names
            if p.requires_grad != want:
                p.requires_grad_(want)

    @contextmanager
    def _ref_scope(self):
        """Within the scope the model computes with the frozen 'ref' adapter (shared base
        weights). ALWAYS use under torch.no_grad(). On exit the policy adapter is restored and
        requires_grad flags are re-asserted (set_adapter(ref) turns ref params' requires_grad
        ON and the policy's OFF in PEFT; set_adapter(policy) reverses it, and we then enforce
        the snapshot explicitly so nothing depends on that behaviour)."""
        if self.ref_adapter is None:
            yield
            return
        self.model.set_adapter(self.ref_adapter)
        try:
            yield
        finally:
            self.model.set_adapter(self._policy_adapter)
            self._enforce_trainable()

    # -- pure: build weighted samples (testable without torch) -------------------
    def _tree_samples(self, problem: Problem) -> list[TrainingSample]:
        import time

        t0 = time.time()
        tree, trajs = self.engine.build_tree(problem, self.cfg.group_size, self.weights)
        samples = build_samples(
            tree,
            trajs,
            problem,
            self.weights,
            base_weight=self.cfg.base_weight,
            max_weight=self.cfg.max_weight,
        )
        rows = [
            {
                "reward": t.reward,
                "success": t.success,
                "n_steps": len(t.steps),
                "premature": t.premature,
                "failed": bool(t.terminal.failed),
                "n_reasoning_steps": t.n_reasoning_steps,
                "bare_verdict": t.ends_bare_verdict,
                "n_proposed": t.n_proposed,
                "n_first_try_valid": t.n_first_try_valid,
                "n_attempts": t.n_attempts,
                "n_degenerate": t.n_degenerate,
            }
            for t in trajs
        ]
        with self._stats_lock:
            self._rollout_rows.extend(rows)
        elapsed = time.time() - t0

        # Count successful trajectories
        success_count = sum(1 for t in trajs if t.success)

        # Get progress info if available (set by build_batch)
        progress_str = ""
        if hasattr(self, "_current_problem_idx") and hasattr(
            self, "_total_problems_in_batch"
        ):
            progress_str = (
                f"[{self._current_problem_idx + 1}/{self._total_problems_in_batch}] "
            )

        # Calculate merge statistics
        merge_pct = (
            (tree.merge_count / tree.state_count * 100) if tree.state_count > 0 else 0
        )

        print(
            f"{progress_str}Problem {problem.id}: {elapsed:.1f}s | "
            f"{success_count}/{len(trajs)} success | {len(samples)} samples | "
            f"{len(tree.nodes)} nodes ({tree.state_count} states, {tree.merge_count} merged = {merge_pct:.0f}%)",
            flush=True,
        )
        return samples

    def _safe_tree_samples(self, problem: Problem) -> list[TrainingSample]:
        """_tree_samples, but an OOM while building ONE problem's tree skips that problem
        (warning + cache flush) instead of crashing the whole run."""
        oom = False
        try:
            return self._tree_samples(problem)
        except Exception as e:  # noqa: BLE001 - re-raised unless it is an OOM
            if not is_oom_error(e):
                raise
            oom = True
            print(f"[grpo_tree] WARNING: OOM while building the tree for problem "
                  f"{problem.id}; skipping this problem ({type(e).__name__})", flush=True)
        finally:
            if oom:
                with self._stats_lock:
                    self._rollout_oom += 1
        _free_cuda_cache()
        return []

    def build_batch(self, problems: list[Problem]) -> list[TrainingSample]:
        import time

        with self._stats_lock:
            self._rollout_rows = []
            self._rollout_oom = 0
        if self.cfg.num_workers and self.cfg.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as ex:
                groups = list(ex.map(self._safe_tree_samples, problems))
        else:
            t_batch_start = time.time()
            groups = []
            for idx, p in enumerate(problems):
                self._current_problem_idx = idx
                self._total_problems_in_batch = len(problems)
                g = self._safe_tree_samples(p)
                groups.append(g)

                # Show ETA after first problem
                if idx == 0:
                    avg_time = time.time() - t_batch_start
                    eta = avg_time * (len(problems) - 1)
                    print(
                        f"  → Batch ETA: {eta:.0f}s ({avg_time:.1f}s/problem)",
                        flush=True,
                    )

        all_samples = [s for g in groups for s in g]
        with self._stats_lock:
            self.last_rollout_stats = aggregate_rollout_stats(self._rollout_rows)
        self.last_rollout_stats["rejected_samples"] = sum(
            1 for s in all_samples if s.meta.get("rejected")
        )
        self.last_rollout_stats["rollout_oom_skipped"] = self._rollout_oom
        return all_samples

    # -- torch update (cluster) --------------------------------------------------
    def _encode(self, messages: list, continuation: str):
        """Tokenize prompt and prompt+continuation on the SAME chat-template rendering the
        policy generated with. Returns `(f_ids[1,L] on CPU, p_len)`, or None when the sequence
        is longer than cfg.max_score_tokens. The prompt span is the longest common token-id
        prefix, so tokenizer merges at the prompt/continuation seam can't misalign targets."""
        tok = self.tokenizer
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # add_special_tokens=False: the template text already contains the special markers;
        # keep both encodings consistent so the prefix actually matches.
        p_ids = (
            tok(prompt_text, return_tensors="pt", add_special_tokens=False)
            .input_ids[0]
            .tolist()
        )
        f_ids = tok(
            prompt_text + continuation, return_tensors="pt", add_special_tokens=False
        ).input_ids
        max_tok = getattr(self.cfg, "max_score_tokens", 0)
        if max_tok and f_ids.shape[1] > max_tok:
            return None
        p_len = _common_prefix_len(p_ids, f_ids[0].tolist())
        return f_ids, p_len

    @staticmethod
    def _forward_mode(causal) -> tuple:
        """How to get logits for ONLY the continuation positions (cached per class):
          ("keep", kwarg)  forward takes logits_to_keep / num_logits_to_keep (int = last N)
          ("decoder", None) run the inner transformer + lm_head on the continuation slice
          ("full", None)   last resort: full logits then slice (old behaviour)"""
        import inspect

        key = type(causal)
        if key in _FWD_MODE_CACHE:
            return _FWD_MODE_CACHE[key]
        try:
            params = inspect.signature(causal.forward).parameters
        except (TypeError, ValueError):
            params = {}
        mode = ("full", None)
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            if kw in params:
                mode = ("keep", kw)
                break
        else:
            prefix = getattr(causal, "base_model_prefix", None)
            if prefix and hasattr(causal, prefix) and hasattr(causal, "lm_head"):
                mode = ("decoder", None)
        _FWD_MODE_CACHE[key] = mode
        return mode

    def _cont_logits(self, model, f_ids, p_len: int):
        """Logits [1, n, V] at positions p_len-1 .. L-2 (those predicting the continuation
        tokens ids[p_len:L]); never materialises [1, L, V] when the model supports it."""
        L = int(f_ids.shape[1])
        start, stop = continuation_span(L, p_len)
        causal = model.get_base_model() if hasattr(model, "get_base_model") else model
        kind, kw = self._forward_mode(causal)
        if kind == "keep":
            out = model(f_ids, use_cache=False, **{kw: logits_to_keep_count(L, p_len)})
            return out.logits[:, :-1, :]  # kept p_len-1..L-1; drop the last row
        if kind == "decoder":
            h = getattr(causal, causal.base_model_prefix)(f_ids, use_cache=False).last_hidden_state
            return causal.lm_head(h[:, start:stop, :])
        return model(f_ids, use_cache=False).logits[:, start:stop, :]

    def _cont_logprobs(self, model, f_ids, p_len: int):
        """Per-token log-probs (1-D, grad-enabled unless under no_grad) of ids[p_len:]."""
        import torch

        f_ids = f_ids.to(model.device)
        if continuation_span(int(f_ids.shape[1]), p_len) is None:  # nothing to score
            return torch.zeros(0, device=model.device)
        logits = self._cont_logits(model, f_ids, p_len)  # [1, n, V]
        targets = f_ids[:, max(p_len, 1):]
        # Memory-efficient token log-probs: log p(t) = logit_t - logsumexp(logits).
        # logsumexp reduces over the vocab dim, so we never materialize a second
        # full [1, n, vocab] tensor the way torch.log_softmax(logits) would (Qwen2.5
        # vocab ~152k -> that copy is several GB on a long rollout and was the OOM).
        sel = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, n]
        tok_logp = (sel - torch.logsumexp(logits, dim=-1))[0]
        del logits, sel
        return tok_logp

    def _token_logprobs(self, model, messages: list, continuation: str):
        """Per-token log-probs of `continuation` under `model` (compat wrapper around
        _encode + _cont_logprobs). None = too long to score; empty tensor = nothing to score."""
        enc = self._encode(messages, continuation)
        if enc is None:
            return None
        f_ids, p_len = enc
        return self._cont_logprobs(model, f_ids, p_len)

    def _seq_logprob(self, model, messages: list, continuation: str):
        """`(sum_logprob, n_continuation_tokens)`; `(None, 0)` = too long to score."""
        tl = self._token_logprobs(model, messages, continuation)
        if tl is None:
            return None, 0
        return tl.sum(), int(tl.shape[0])

    def _ensure_opt(self) -> None:
        import torch

        if hasattr(self, "_opt"):
            return
        named = list(self.model.named_parameters())
        keep = set(trainable_param_names([(n, p.requires_grad) for n, p in named],
                                         self.ref_adapter))
        self._params = [p for n, p in named if n in keep]
        if not self._params:
            raise RuntimeError("no trainable parameters for the optimizer")
        if self.ref_adapter is not None:
            leaked = [n for n, p in named if p.requires_grad and n not in keep]
            if leaked:
                raise RuntimeError(f"ref adapter params require grad: {leaked[:3]}")
        print(f"[grpo_tree] optimizer over {len(self._params)} tensors "
              f"({sum(p.numel() for p in self._params)/1e6:.1f}M params; "
              f"policy adapter={self._policy_adapter!r}, ref adapter={self.ref_adapter!r})",
              flush=True)
        self._opt = torch.optim.AdamW(self._params, lr=self.cfg.lr)

    def _score_sample(self, s: TrainingSample, adv: float, n_batch: int):
        """Score + back-prop ONE sample. Order: frozen-reference forward first (no grad), then
        policy forward + backward, so peak memory is the policy pass alone. Returns None if
        the sample is skipped (too long / nothing to score), else a dict of scalars. Kept as
        its own frame so every tensor is released on return (or unwind after an OOM)."""
        import torch

        enc = self._encode(s.messages, s.continuation)
        if enc is None:
            return None
        f_ids, p_len = enc
        n_tok = int(f_ids.shape[1]) - max(p_len, 1)
        if n_tok <= 0:
            return None

        rtl = None
        if self._kl_enabled:
            with torch.no_grad():
                if self.ref_adapter is not None:
                    with self._ref_scope():  # same device/base as the policy
                        rtl = self._cont_logprobs(self.model, f_ids, p_len)
                else:
                    rtl = self._cont_logprobs(self.ref_model, f_ids, p_len)
            rtl = rtl.detach()

        tl = self._cont_logprobs(self.model, f_ids, p_len)
        if tl.shape[0] == 0:
            return None
        denom = loss_denominator(self.cfg.loss_norm, n_tok, self.cfg.loss_norm_const)
        kl_sum = None
        kl_mean_tok = 0.0
        if rtl is not None and rtl.shape[0] == n_tok:
            # per-token k3 estimator of KL(policy || ref) on the sampled tokens; its gradient
            # w.r.t. the policy is (1 - exp(r_t - l_t)) * grad l_t, which pulls the policy
            # toward the reference.
            kl_tok = k3_kl_tokens(tl, rtl.to(tl.device))
            kl_sum = kl_tok.sum()
            kl_mean_tok = float(kl_tok.detach().mean())
        loss_s, pg_s, kl_s = sample_loss(
            adv * s.weight, tl.sum(), kl_sum, self.cfg.kl_coef, denom
        )
        (loss_s / n_batch).backward()
        return {
            "loss": float(loss_s.detach()),
            "pg": float(pg_s.detach()),
            "kl": kl_mean_tok,
            "n_tok": n_tok,
        }

    def _update(self, samples: list[TrainingSample]) -> dict:
        import torch

        self._ensure_opt()
        # normalise advantages for stability (mode: cfg.adv_whiten; see loss.whiten_advantages)
        wadvs = whiten_advantages([s.advantage for s in samples], self.cfg.adv_whiten)

        # Activation checkpointing: recompute layer activations during backward instead of
        # storing them all. This is the big memory lever for a single-sequence forward
        # through the 7B policy. Enable only for the update; generation (build_batch) ran
        # earlier and wants use_cache=True. use_reentrant=False + enable_input_require_grads
        # are both required for a frozen-base PEFT model, otherwise backward has no input
        # that requires grad and the checkpointed segment errors. (The no-grad ref forward is
        # unaffected: checkpointing is a no-op without autograd.)
        gc_on = getattr(self.cfg, "grad_checkpointing", False)
        if gc_on:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.config.use_cache = False
            # HF only checkpoints when `self.training`; train() the policy but keep every
            # dropout (incl. LoRA dropout) off so the forward stays deterministic.
            self.model.train()
            for m in dropout_modules_to_eval(self.model):
                m.eval()
        try:
            return self._update_body(samples, wadvs)
        finally:
            if gc_on:
                self.model.gradient_checkpointing_disable()
                self.model.config.use_cache = True
                self.model.eval()

    def _update_body(self, samples: list[TrainingSample], wadvs) -> dict:
        import torch

        self._opt.zero_grad()
        run_loss = run_pg = run_kl = 0.0
        tok_total = 0

        # Score one sample at a time and back-prop immediately so each sample's autograd
        # graph is freed before the next. (Accumulating loss across the batch would keep
        # every graph alive at once -> OOM.) Skipped samples don't count toward the mean.
        scored = 0
        n_oom = 0
        n = max(len(samples), 1)
        empty_cache_every = 8  # release fragmented blocks periodically
        for i, s in enumerate(samples):
            oom = False
            r = None
            try:
                r = self._score_sample(s, wadvs[i], n)
            except Exception as e:  # noqa: BLE001 - re-raised unless OOM
                if not is_oom_error(e):
                    raise
                oom = True
                print(f"[grpo_tree] WARNING: OOM scoring sample {i + 1}/{len(samples)}; "
                      f"skipping it (grads accumulated so far are kept)", flush=True)
            if oom:  # outside the except block so the traceback no longer pins tensors
                n_oom += 1
                _free_cuda_cache()
                continue
            if r is None:  # too long, or nothing to score
                continue
            run_loss += r["loss"]
            run_pg += r["pg"]
            run_kl += r["kl"]
            tok_total += r["n_tok"]
            scored += 1
            if (i + 1) % empty_cache_every == 0:
                torch.cuda.empty_cache()

        oom_step_skipped = oom_skip_step(n_oom, len(samples), self.cfg.max_oom_skip_frac)
        grad_norm = 0.0
        if oom_step_skipped:
            print(f"[grpo_tree] {n_oom}/{len(samples)} samples OOMed (> "
                  f"max_oom_skip_frac={self.cfg.max_oom_skip_frac}); skipping optimizer step",
                  flush=True)
            skipped = True
            self._opt.zero_grad()
        else:
            if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(self._params, self.cfg.max_grad_norm)
                )
            else:
                grad_norm = float(
                    torch.norm(
                        torch.stack(
                            [p.grad.norm() for p in self._params if p.grad is not None]
                            or [torch.zeros(())]
                        )
                    )
                )
            skipped = not math.isfinite(grad_norm)
            if skipped:  # never step on NaN/inf grads: it would corrupt the weights
                print(f"[grpo_tree] non-finite grad_norm ({grad_norm}); skipping optimizer step",
                      flush=True)
                self._opt.zero_grad()
            else:
                self._opt.step()
        torch.cuda.empty_cache()

        self._enforce_trainable()  # belt and braces: ONLY the policy adapter trains

        d = max(scored, 1)
        return {
            "loss": run_loss / d,
            "pg_loss": run_pg / d,
            "kl": run_kl / d,  # mean per-token k3 KL(policy || ref); 0 if no ref model
            "grad_norm": grad_norm,  # pre-clip global norm
            "mean_cont_tokens": tok_total / d,
            "n_samples": scored,
            "n_total": len(samples),
            "skipped_step": skipped,
            "oom_skipped": n_oom,
            "oom_step_skipped": oom_step_skipped,
        }

    # -- memory telemetry --------------------------------------------------------
    def _reset_peak_mem(self) -> None:
        if not self.mem_devices:
            return
        import torch

        for i in self.mem_devices:
            torch.cuda.reset_peak_memory_stats(i)

    def _peak_mem(self) -> dict:
        """Flat {'peak_mem_gib_cuda<i>': GiB} of max_memory_allocated since the last reset."""
        if not self.mem_devices:
            return {}
        import torch

        return format_peak_mem({i: torch.cuda.max_memory_allocated(i) for i in self.mem_devices})

    def _metrics_record(self, step: int, update_stats: dict) -> dict:
        r = self.last_rollout_stats
        return {
            "step": step,
            "loss": update_stats.get("loss"),
            "pg_loss": update_stats.get("pg_loss"),
            "kl": update_stats.get("kl"),
            "grad_norm": update_stats.get("grad_norm"),
            "n_samples": update_stats.get("n_samples"),
            "mean_reward": r["mean_reward"],
            "success_rate": r["success_rate"],
            "mean_steps_per_rollout": r["mean_steps"],
            "frac_premature_verdict": r["premature_frac"],  # verdict with < min_steps steps
            "frac_failed": r["failed_frac"],
            "bare_verdict_frac": r.get("bare_verdict_frac"),  # terminal step is a bare "Verdict:" line
            "mean_cont_tokens": update_stats.get("mean_cont_tokens"),
            "first_try_valid_rate": r.get("first_try_valid_rate"),
            "rejected_samples": r.get("rejected_samples"),
            "mean_reasoning_steps": r.get("mean_reasoning_steps"),  # successful rollouts only
            "degenerate_frac": r.get("degenerate_frac"),  # empty / repetitive proposed steps
            "oom_skipped": update_stats.get("oom_skipped", 0),  # samples skipped on OOM in scoring
            "rollout_oom_skipped": r.get("rollout_oom_skipped", 0),  # problems skipped on OOM in rollout
            "oom_step_skipped": update_stats.get("oom_step_skipped", False),
        }

    @staticmethod
    def _append_metrics(output_dir: str, record: dict) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train_metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")

    # -- checkpointing / resume --------------------------------------------------
    def _save_adapter(self, path: str) -> None:
        """Save ONLY the policy weights (never the frozen 'ref' adapter)."""
        if self.ref_adapter is None:
            self.model.save_pretrained(path)
            return
        try:
            self.model.save_pretrained(path, selected_adapters=[self._policy_adapter])
        except TypeError:  # very old peft without selected_adapters: save all, drop ref
            import shutil

            self.model.save_pretrained(path)
            shutil.rmtree(os.path.join(path, self.ref_adapter), ignore_errors=True)

    @staticmethod
    def _capture_rng() -> dict:
        import random

        out: dict = {"python": encode_py_random_state(random.getstate())}
        try:
            import numpy as np

            kind, keys, pos, has_gauss, cached = np.random.get_state()
            out["numpy"] = [kind, keys.tolist(), int(pos), int(has_gauss), float(cached)]
        except Exception:  # noqa: BLE001
            pass
        try:
            import torch

            out["torch"] = torch.get_rng_state().tolist()
            if torch.cuda.is_available():
                out["torch_cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
        except Exception:  # noqa: BLE001
            pass
        return out

    @staticmethod
    def _restore_rng(rng: dict) -> None:
        import random

        if not rng:
            return
        if "python" in rng:
            random.setstate(decode_py_random_state(rng["python"]))
        try:
            if "numpy" in rng:
                import numpy as np

                kind, keys, pos, has_gauss, cached = rng["numpy"]
                np.random.set_state((kind, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
        except Exception as e:  # noqa: BLE001
            print(f"[grpo_tree] could not restore numpy RNG: {e}", flush=True)
        try:
            import torch

            if "torch" in rng:
                torch.set_rng_state(torch.tensor(rng["torch"], dtype=torch.uint8))
            if "torch_cuda" in rng and torch.cuda.is_available():
                n = min(len(rng["torch_cuda"]), torch.cuda.device_count())
                for i in range(n):
                    torch.cuda.set_rng_state(torch.tensor(rng["torch_cuda"][i], dtype=torch.uint8), i)
        except Exception as e:  # noqa: BLE001
            print(f"[grpo_tree] could not restore torch RNG: {e}", flush=True)

    def _trainer_state(self) -> TrainerState:
        return TrainerState(
            step=self._step,
            problem_index=self._problem_index,
            epoch=self._epoch,
            n_problems=self._n_problems,
            batch_problems=self.cfg.batch_problems,
            early_stop=self.stopper.state_dict(),
            last_healthy_step=self.last_healthy_step,
            last_healthy_path=self.last_healthy_path,
            rng=self._capture_rng(),
        )

    def _save(self, output_dir: str, tag: str) -> str:
        """Policy adapter + tokenizer + optimizer state + trainer_state.json (everything
        `resume_from` needs). Returns the checkpoint dir."""
        path = os.path.join(output_dir, tag)
        self._save_adapter(path)
        self.tokenizer.save_pretrained(path)
        if hasattr(self, "_opt"):
            import torch

            torch.save(self._opt.state_dict(), os.path.join(path, OPTIMIZER_FILE))
        save_trainer_state(os.path.join(path, TRAINER_STATE_FILE), self._trainer_state())
        print(f"[grpo_tree] checkpoint saved -> {path}")
        return path

    def save_final(self, output_dir: str) -> None:
        """Final policy adapter + tokenizer into output_dir (policy only; no 'ref' adapter)."""
        self._save_adapter(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def _resume(self, path: str, output_dir: Optional[str]) -> TrainerState:
        """Restore adapter weights (policy adapter only), optimizer state, trainer state,
        early-stop state and RNGs from a `checkpoint-N` dir; drop metrics lines past N."""
        if not hasattr(self.model, "peft_config"):
            raise NotImplementedError("resume_from currently supports PEFT (LoRA) policies only")
        ts_path = os.path.join(path, TRAINER_STATE_FILE)
        if not os.path.exists(ts_path):
            raise FileNotFoundError(
                f"{ts_path} missing: '{path}' is not a resumable checkpoint (written by this "
                f"trainer version?). Use a checkpoint-N / checkpoint-latest dir.")
        state = load_trainer_state(ts_path)
        from peft import load_peft_weights, set_peft_model_state_dict

        print(f"[grpo_tree] RESUME from {path}: step={state.step} "
              f"problem_index={state.problem_index} epoch={state.epoch}", flush=True)
        weights = load_peft_weights(path)
        set_peft_model_state_dict(self.model, weights, adapter_name=self._policy_adapter)
        del weights
        self._enforce_trainable()
        self._ensure_opt()
        opt_path = os.path.join(path, OPTIMIZER_FILE)
        if os.path.exists(opt_path):
            import torch

            self._opt.load_state_dict(torch.load(opt_path, map_location="cpu"))
        else:
            print(f"[grpo_tree] WARNING: {opt_path} missing -> optimizer state starts fresh",
                  flush=True)
        self.stopper.load_state_dict(state.early_stop)
        self.last_healthy_step = state.last_healthy_step
        self.last_healthy_path = state.last_healthy_path
        self._restore_rng(state.rng)
        if output_dir:
            dropped = truncate_metrics_after(os.path.join(output_dir, "train_metrics.jsonl"),
                                             state.step)
            if dropped:
                print(f"[grpo_tree] dropped {dropped} train_metrics.jsonl record(s) past step "
                      f"{state.step} (backup: train_metrics.jsonl.pre-resume.bak)", flush=True)
        return state

    def _handle_decision(self, d, output_dir: Optional[str]) -> bool:
        """Log an early-stop decision. Returns True if training should stop."""
        tag = "EARLY STOP" if d.stop else "EARLY-STOP WARNING (mode=warn, continuing)"
        print(f"\n[grpo_tree] {tag} at step {d.step}: {d.reason}", flush=True)
        if not d.stop:
            return False
        self.early_stop_info = {"early_stop": True, "reason": d.reason, "step": d.step}
        healthy = self.stopper.best_ok
        print(
            f"[grpo_tree] last healthy step: {healthy}"
            + (f" (last-healthy checkpoint is from step {self.last_healthy_step}: "
               f"{self.last_healthy_path})" if self.last_healthy_path else
               " (no last-healthy checkpoint was saved)"),
            flush=True,
        )
        if output_dir:
            self._append_metrics(output_dir, dict(self.early_stop_info))
            self._save(output_dir, f"early-stop-{d.step}")
        return True

    def train(
        self,
        problems: list[Problem],
        output_dir: Optional[str] = None,
        save_every: int = 0,
        resume_from: Optional[str] = None,
    ) -> list[dict]:
        """Run training. If output_dir + save_every>0 are given, write a checkpoint every
        `save_every` updates (checkpoint-N plus a rolling checkpoint-latest) so a SLURM
        wall-time kill / preemption does not lose the run. `resume_from` = a checkpoint dir:
        weights, optimizer, early-stop state and RNGs are restored and already-processed
        problems are skipped."""
        import time

        # eval() mode: disable dropout (incl. LoRA dropout) so both generation and the
        # policy-gradient log-probs are deterministic given the sampled tokens. Autograd
        # still works in eval mode; we only sample via temperature, not dropout.
        self.model.eval()
        if self.ref_model is not None:
            self.ref_model.eval()

        stats = []
        stopped = False
        bp = self.cfg.batch_problems
        n_prob = len(problems)
        self._n_problems = n_prob
        step = 0
        start_epoch, start_offset = 0, 0
        if resume_from:
            st = self._resume(resume_from, output_dir)
            step = self._step = st.step
            self._problem_index = st.problem_index
            if st.n_problems and st.n_problems != n_prob:
                print(f"[grpo_tree] WARNING: problem set size changed since the checkpoint "
                      f"({st.n_problems} -> {n_prob}); resume position may be off", flush=True)
            start_epoch, start_offset = resume_plan(st.problem_index, n_prob, self.cfg.epochs)
            print(f"[grpo_tree] resuming at epoch {start_epoch + 1}, problem offset "
                  f"{start_offset} (skipping {st.problem_index} processed problems)", flush=True)
        total_steps = (n_prob + bp - 1) // bp * self.cfg.epochs
        start_step = step
        self._train_start_time = time.time()

        print(f"\n{'='*60}")
        print("Starting RL Training")
        print(f"{'='*60}")
        print(
            f"Total problems: {n_prob} | Batch size: {bp} | Epochs: {self.cfg.epochs}"
        )
        print(f"Total gradient steps: {total_steps}")
        print(f"{'='*60}\n", flush=True)

        for epoch in range(start_epoch, self.cfg.epochs):
            if stopped:
                break
            self._epoch = epoch
            first = start_offset if epoch == start_epoch else 0
            for i in range(first, n_prob, bp):
                batch_end = min(i + bp, n_prob)
                batch_problems = problems[i:batch_end]

                print(
                    f"\n>>> Step {step + 1}/{total_steps} | Problems {i + 1}-{batch_end}/{n_prob}",
                    flush=True,
                )

                self._reset_peak_mem()  # peaks below cover rollouts + update of THIS step
                t0 = time.time()
                samples = self.build_batch(batch_problems)
                t1 = time.time()
                self._problem_index = epoch * n_prob + batch_end

                if samples:
                    update_stats = self._update(samples)
                    t2 = time.time()

                    avg_step_time = (time.time() - self._train_start_time) / (step - start_step + 1)
                    remaining_steps = total_steps - (step + 1)
                    eta_seconds = avg_step_time * remaining_steps
                    eta_minutes = eta_seconds / 60

                    print(
                        f"  Rollouts: {t1 - t0:.1f}s | Update: {t2 - t1:.1f}s | "
                        f"Loss: {update_stats.get('loss', 0):.4f} | Samples: {len(samples)}",
                        flush=True,
                    )
                    print(
                        f"  ETA: {eta_minutes:.1f} min ({avg_step_time:.1f}s/step)\n",
                        flush=True,
                    )

                    stats.append(update_stats)
                    step += 1
                    self._step = step
                    record = self._metrics_record(step, update_stats)
                    record.update(self._peak_mem())
                    if output_dir:
                        self._append_metrics(output_dir, record)
                    decision = self.stopper.update(record)
                    if decision is not None and self._handle_decision(decision, output_dir):
                        stopped = True
                        break
                    if save_every and output_dir and step % save_every == 0:
                        ckpt = self._save(output_dir, f"checkpoint-{step}")
                        publish_copy(ckpt, os.path.join(output_dir, "checkpoint-latest"))
                        if self.stopper.last_healthy:
                            lh = os.path.join(output_dir, "last-healthy")
                            publish_copy(ckpt, lh)
                            self.last_healthy_step = step
                            self.last_healthy_path = lh
                else:
                    print("  WARNING: No samples generated!\n", flush=True)
                    decision = self.stopper.update({
                        "step": step, "n_samples": 0,
                        "rollout_oom_skipped": self.last_rollout_stats.get("rollout_oom_skipped", 0),
                    })
                    if decision is not None and self._handle_decision(decision, output_dir):
                        stopped = True
                        break

        print(f"\n{'='*60}")
        if stopped:
            print(f"Training EARLY-STOPPED at step {step}: {self.early_stop_info['reason']}")
        else:
            print(f"Training Complete! {step} steps finished.")
        print(f"{'='*60}\n", flush=True)
        return stats


def _load_judge_model(rl_cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = rl_cfg.get("judge_model", "Qwen/Qwen2.5-7B-Instruct")

    # Pin the judge to its own device so it does NOT share the policy's GPU. Default is
    # cuda:1 (a second allocated GPU) which frees ~5GB on the policy card. Set judge_device
    # to "cuda:0" to co-locate on one GPU again, or "cpu" if you only have a single GPU.
    # For 4-bit (bitsandbytes) the placement must be expressed as a device_map, not .to().
    judge_device = rl_cfg.get("judge_device", "cuda:1")
    dev_index = judge_device
    if isinstance(judge_device, str) and judge_device.startswith("cuda:"):
        dev_index = int(judge_device.split(":", 1)[1])
    elif judge_device == "cuda":
        dev_index = 0

    kw = {"torch_dtype": "auto", "device_map": {"": dev_index}}
    if rl_cfg.get("judge_load_in_4bit"):
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    print(
        f"[grpo_tree] loading judge on device {judge_device} (device_map={{'': {dev_index!r}}})",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def _assert_single_process(output_dir: str) -> None:
    """grpo_tree is single-process (hand-rolled optimizer, not DDP/FSDP). If launched with
    multiple tasks (srun -n>1 / torchrun) each rank would train a full copy and clobber the
    same output_dir. Fail loudly instead."""
    world = int(os.environ.get("WORLD_SIZE") or os.environ.get("SLURM_NTASKS") or "1")
    if world > 1:
        rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID") or "?"
        raise RuntimeError(
            f"grpo_tree is single-process but was launched with "
            f"WORLD_SIZE/SLURM_NTASKS={world} (rank={rank}). Multiple ranks would each "
            f"train a full copy and clobber '{output_dir}'. Launch with a single task "
            f"(srun -n1, no torchrun), or use method=grpo_baseline for multi-GPU DDP."
        )


def _load_ref_model(rl_cfg: dict, model_config: Optional[dict]):
    """Load a FROZEN copy of the starting policy for the KL term.

    The trainer only applies KL when `ref_model is not None`; if we don't build one here
    the penalty is silently skipped and the policy is free to drift off the SFT
    distribution (which is what let it collapse to bare `boxed{X}` / comma spam). The
    reference is the checkpoint the policy was initialised from, so KL anchors us to SFT
    — NOT to the base model (disabling the LoRA adapter would give the base, which is
    the wrong target).
    """
    if not rl_cfg.get("use_kl", True):
        print("[grpo_tree] use_kl=false -> training WITHOUT a KL penalty (not recommended)")
        return None

    model_config = model_config or {}
    path = rl_cfg.get("ref_model_path") or model_config.get("from_checkpoint")
    if not path:
        print(
            "[grpo_tree] WARNING: no reference checkpoint found "
            "(set rl.ref_model_path or model.from_checkpoint) -> KL DISABLED"
        )
        return None

    from transformers import AutoModelForCausalLM
    from peft import AutoPeftModelForCausalLM

    device = rl_cfg.get("ref_device")
    if not device:
        device = "cpu"
        print(
            "[grpo_tree] WARNING: rl.ref_device not set -> KL reference on CPU (slow). "
            "Set rl.ref_device (e.g. 'cuda:1') to speed it up."
        )

    is_peft = model_config.get("is_peft_checkpoint", False)
    dtype = model_config.get("torch_dtype", "bfloat16")
    print(f"Loading frozen KL reference from {path} (device={device})...", flush=True)
    loader = AutoPeftModelForCausalLM if is_peft else AutoModelForCausalLM
    ref = loader.from_pretrained(path, torch_dtype=dtype, device_map={"": device})
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


def _setup_kl_reference(model, rl_cfg: dict, model_config: Optional[dict]):
    """-> (ref_model, ref_adapter). rl.ref_mode:
      "adapter"  (default) load the SAME PEFT checkpoint the policy started from a second
                 time onto the policy as a frozen adapter named "ref": zero extra base-model
                 memory (only the tiny LoRA weights), ref forward runs on the policy's GPU.
      "separate" old behaviour: full frozen copy on rl.ref_device (~15 GiB for a bf16 7B).
    Falls back to "separate" (with a warning) when the policy is not a PEFT model."""
    mode = parse_ref_mode(rl_cfg.get("ref_mode"))
    if not rl_cfg.get("use_kl", True):
        return _load_ref_model(rl_cfg, model_config), None  # prints the use_kl=false notice
    model_config = model_config or {}
    path = rl_cfg.get("ref_model_path") or model_config.get("from_checkpoint")
    has_adapter = bool(path) and (
        not os.path.isdir(path) or os.path.exists(os.path.join(path, "adapter_config.json"))
    )  # a non-local path (hub id) is assumed to be an adapter repo
    mode, warning = resolve_ref_mode(mode, hasattr(model, "peft_config"), has_adapter)
    if warning:
        print(f"[grpo_tree] WARNING: {warning}", flush=True)
    if mode == "separate":
        return _load_ref_model(rl_cfg, model_config), None
    if not path:
        print("[grpo_tree] WARNING: no reference checkpoint found "
              "(set rl.ref_model_path or model.from_checkpoint) -> KL DISABLED")
        return None, None
    print(f"[grpo_tree] KL reference = frozen adapter '{REF_ADAPTER_NAME}' loaded from {path} "
          f"onto the policy (shared base weights; ref_mode=adapter)", flush=True)
    model.load_adapter(path, adapter_name=REF_ADAPTER_NAME, is_trainable=False)
    return None, REF_ADAPTER_NAME


def run_grpo_tree(model, tokenizer, rl_cfg: dict, model_config: Optional[dict] = None):
    """Entry from run.py. In-process policy (the trained model) + in-process judge."""
    _assert_single_process(rl_cfg["output_dir"])

    from .embedder import build_state_matcher

    policy = TransformersPolicy(
        model, tokenizer, temperature=rl_cfg.get("temperature", 0.8)
    )
    print(
        f"Loading judge model: {rl_cfg.get('judge_model', 'Qwen/Qwen2.5-7B-Instruct')}...",
        flush=True,
    )
    jm, jt = _load_judge_model(rl_cfg)
    judge = TransformersJudge(jm, jt)
    matcher = build_state_matcher(
        rl_cfg.get("merge", {})
    )  # semantic merging by default

    ref_model, ref_adapter = _setup_kl_reference(model, rl_cfg, model_config)
    mem_devices = cuda_indices(
        getattr(model, "device", None),
        rl_cfg.get("ref_device") if ref_model is not None else None,
        rl_cfg.get("judge_device", "cuda:1"),
        (rl_cfg.get("merge") or {}).get("device") if (rl_cfg.get("merge") or {}).get("semantic", True) else None,
    )

    problems = load_problems_jsonl(
        rl_cfg["problems_path"], limit=rl_cfg.get("max_problems")
    )

    cfg = TreeGRPOConfig(**rl_cfg.get("tree", {}))
    weights = RewardWeights.from_dict(rl_cfg.get("reward"))
    print(f"[grpo_tree] effective RewardWeights: {asdict(weights)}", flush=True)
    print(f"[grpo_tree] effective TreeGRPOConfig: {asdict(cfg)}", flush=True)
    early_cfg = EarlyStopConfig.from_dict(rl_cfg.get("early_stop"))
    print(f"[grpo_tree] effective EarlyStopConfig: {asdict(early_cfg)}", flush=True)
    trainer = GRPOTreeTrainer(
        model,
        tokenizer,
        policy,
        judge,
        ref_model=ref_model,
        cfg=cfg,
        matcher=matcher,
        weights=weights,
        early_stop=early_cfg,
        ref_adapter=ref_adapter,
        mem_devices=mem_devices,
    )
    stats = trainer.train(
        problems,
        output_dir=rl_cfg["output_dir"],
        save_every=rl_cfg.get("save_every", 0),
        resume_from=rl_cfg.get("resume_from"),
    )
    trainer.save_final(rl_cfg["output_dir"])  # policy adapter only (never the frozen "ref")
    if trainer.early_stop_info:
        print(
            f"\n[grpo_tree] EARLY-STOPPED at step {trainer.early_stop_info['step']}: "
            f"{trainer.early_stop_info['reason']}\n"
            f"[grpo_tree] final adapters saved to {rl_cfg['output_dir']} (state at stop; "
            f"last healthy checkpoint: {trainer.last_healthy_path or 'none saved'}, "
            f"healthy step {trainer.stopper.best_ok})"
        )
    print(
        f"\n[grpo_tree] Complete! {len(stats)} updates, final loss: {stats[-1]['loss']:.4f}"
        if stats
        else "\n[grpo_tree] Complete!"
    )

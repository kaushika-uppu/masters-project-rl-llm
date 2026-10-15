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
from dataclasses import asdict, dataclass
from typing import Optional

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
    ):
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
        self.cfg = cfg
        self.weights = weights
        self.engine = RolloutEngine(
            policy, judge, matcher, cfg.max_depth, cfg.retry_budget
        )

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

    def build_batch(self, problems: list[Problem]) -> list[TrainingSample]:
        import time

        with self._stats_lock:
            self._rollout_rows = []
        if self.cfg.num_workers and self.cfg.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as ex:
                groups = list(ex.map(self._tree_samples, problems))
        else:
            t_batch_start = time.time()
            groups = []
            for idx, p in enumerate(problems):
                self._current_problem_idx = idx
                self._total_problems_in_batch = len(problems)
                g = self._tree_samples(p)
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
        return all_samples

    # -- torch update (cluster) --------------------------------------------------
    def _token_logprobs(self, model, messages: list, continuation: str):
        """Per-token log-probs (1-D tensor, grad-enabled unless called under no_grad) of
        `continuation` under `model`, scored on the SAME chat-template rendering the policy
        generated with. The prompt span is found by longest-common-prefix of token ids, so
        tokenizer merges at the prompt/continuation seam can't misalign the target.

        Returns None when the sequence is too long to score safely (max_score_tokens) so
        the caller can skip the sample. Returns an empty tensor if there is nothing to score."""
        import torch

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
        f_ids = f_ids.to(model.device)
        p_len = _common_prefix_len(p_ids, f_ids[0].tolist())
        if f_ids.shape[1] <= p_len:  # nothing to score
            return torch.zeros(0, device=model.device)
        logits = model(f_ids).logits[:, :-1, :]
        targets = f_ids[:, 1:]
        # Memory-efficient token log-probs: log p(t) = logit_t - logsumexp(logits).
        # logsumexp reduces over the vocab dim, so we never materialize a second
        # full [1, L, vocab] tensor the way torch.log_softmax(logits) would (Qwen2.5
        # vocab ~152k -> that copy is several GB on a long rollout and was the OOM).
        sel = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [1, L-1]
        tok_logp = (sel - torch.logsumexp(logits, dim=-1))[0]
        del logits, sel
        return tok_logp[p_len - 1 :]

    def _seq_logprob(self, model, messages: list, continuation: str):
        """`(sum_logprob, n_continuation_tokens)`; `(None, 0)` = too long to score."""
        tl = self._token_logprobs(model, messages, continuation)
        if tl is None:
            return None, 0
        return tl.sum(), int(tl.shape[0])

    def _update(self, samples: list[TrainingSample]) -> dict:
        import torch

        if not hasattr(self, "_opt"):
            self._params = [p for p in self.model.parameters() if p.requires_grad]
            self._opt = torch.optim.AdamW(self._params, lr=self.cfg.lr)
        # normalise advantages for stability (mode: cfg.adv_whiten; see loss.whiten_advantages)
        wadvs = whiten_advantages([s.advantage for s in samples], self.cfg.adv_whiten)

        # Activation checkpointing: recompute layer activations during backward instead of
        # storing them all. This is the big memory lever for a single-sequence forward
        # through the 7B policy. Enable only for the update; generation (build_batch) ran
        # earlier and wants use_cache=True. use_reentrant=False + enable_input_require_grads
        # are both required for a frozen-base PEFT model, otherwise backward has no input
        # that requires grad and the checkpointed segment errors.
        gc_on = getattr(self.cfg, "grad_checkpointing", False)
        if gc_on:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.config.use_cache = False

        self._opt.zero_grad()
        run_loss = run_pg = run_kl = 0.0
        tok_total = 0

        # Score one sample at a time and back-prop immediately so each sample's autograd
        # graph is freed before the next. (Accumulating loss across the batch would keep
        # every graph alive at once -> OOM.) Skipped samples don't count toward the mean.
        scored = 0
        n = max(len(samples), 1)
        empty_cache_every = 8  # release fragmented blocks periodically
        for i, s in enumerate(samples):
            adv = wadvs[i]
            tl = self._token_logprobs(self.model, s.messages, s.continuation)
            if tl is None or tl.shape[0] == 0:  # too long, or nothing to score
                continue
            n_tok = int(tl.shape[0])
            denom = loss_denominator(
                self.cfg.loss_norm, n_tok, self.cfg.loss_norm_const
            )
            kl_sum = None
            kl_mean_tok = 0.0
            if self.ref_model is not None:
                with torch.no_grad():
                    rtl = self._token_logprobs(self.ref_model, s.messages, s.continuation)
                if rtl is not None and rtl.shape[0] == n_tok:
                    # per-token k3 estimator of KL(policy || ref) on the sampled tokens;
                    # its gradient w.r.t. the policy is (1 - exp(r_t - l_t)) * grad l_t,
                    # which actually pulls the policy toward the reference.
                    kl_tok = k3_kl_tokens(tl, rtl.to(tl.device))
                    kl_sum = kl_tok.sum()
                    kl_mean_tok = float(kl_tok.detach().mean())
            loss_s, pg_s, kl_s = sample_loss(
                adv * s.weight, tl.sum(), kl_sum, self.cfg.kl_coef, denom
            )
            (loss_s / n).backward()
            run_loss += float(loss_s.detach())
            run_pg += float(pg_s.detach())
            run_kl += kl_mean_tok
            tok_total += n_tok
            scored += 1
            del loss_s, pg_s, kl_s, tl, kl_sum
            if (i + 1) % empty_cache_every == 0:
                torch.cuda.empty_cache()

        grad_norm = 0.0
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

        if gc_on:
            self.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True

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
        }

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
        }

    @staticmethod
    def _append_metrics(output_dir: str, record: dict) -> None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train_metrics.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")

    def _save(self, output_dir: str, tag: str) -> None:
        path = os.path.join(output_dir, tag)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[grpo_tree] checkpoint saved -> {path}")

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
    ) -> list[dict]:
        """Run training. If output_dir + save_every>0 are given, write a checkpoint every
        `save_every` updates so a SLURM wall-time kill / preemption does not lose the run
        (resume by pointing the config's from_checkpoint at the latest checkpoint dir)."""
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
        step = 0
        total_steps = (len(problems) + bp - 1) // bp * self.cfg.epochs

        print(f"\n{'='*60}")
        print("Starting RL Training")
        print(f"{'='*60}")
        print(
            f"Total problems: {len(problems)} | Batch size: {bp} | Epochs: {self.cfg.epochs}"
        )
        print(f"Total gradient steps: {total_steps}")
        print(f"{'='*60}\n", flush=True)

        for _ in range(self.cfg.epochs):
            if stopped:
                break
            for i in range(0, len(problems), bp):
                batch_end = min(i + bp, len(problems))
                batch_problems = problems[i:batch_end]

                print(
                    f"\n>>> Step {step + 1}/{total_steps} | Problems {i + 1}-{batch_end}/{len(problems)}",
                    flush=True,
                )

                t0 = time.time()
                samples = self.build_batch(batch_problems)
                t1 = time.time()

                if samples:
                    update_stats = self._update(samples)
                    t2 = time.time()

                    # Calculate ETA
                    if step == 0:
                        self._train_start_time = t0
                    avg_step_time = (time.time() - self._train_start_time) / (step + 1)
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
                    record = self._metrics_record(step, update_stats)
                    if output_dir:
                        self._append_metrics(output_dir, record)
                    decision = self.stopper.update(record)
                    if decision is not None and self._handle_decision(decision, output_dir):
                        stopped = True
                        break
                    if save_every and output_dir and step % save_every == 0:
                        self._save(output_dir, f"checkpoint-{step}")
                        if self.stopper.last_healthy:
                            self._save(output_dir, "last-healthy")
                            self.last_healthy_step = step
                            self.last_healthy_path = os.path.join(output_dir, "last-healthy")
                else:
                    print("  WARNING: No samples generated!\n", flush=True)
                    decision = self.stopper.update({"step": step, "n_samples": 0})
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

    ref_model = _load_ref_model(rl_cfg, model_config)

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
    )
    stats = trainer.train(
        problems,
        output_dir=rl_cfg["output_dir"],
        save_every=rl_cfg.get("save_every", 0),
    )
    model.save_pretrained(rl_cfg["output_dir"])
    tokenizer.save_pretrained(rl_cfg["output_dir"])
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

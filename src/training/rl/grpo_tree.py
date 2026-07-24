"""Tree-GRPO trainer: per-step advantages from the rollout tree (no value network).

Flow per train step:
  problems -> RolloutEngine builds a tree per problem (step-by-step gen + judge + retry)
           -> Monte-Carlo node values (pooled terminal rewards)
           -> build_samples: per-step (advantage = MC adv + novelty + redundancy,
              weight = base + criticality)
           -> policy-gradient update: loss = -(adv*weight)*logprob(step) + kl*KL(model||ref)

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

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from .inproc import TransformersJudge, TransformersPolicy
from .merge import StateMatcher
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
    num_workers: int = 1  # parallel trees across problems (see module note)


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
        weights: RewardWeights = RewardWeights(),
    ):
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
            tree, trajs, problem, self.weights, base_weight=self.cfg.base_weight
        )
        elapsed = time.time() - t0

        # Count successful trajectories
        success_count = sum(1 for t in trajs if t.success)

        # Get progress info if available (set by build_batch)
        progress_str = ""
        if hasattr(self, '_current_problem_idx') and hasattr(self, '_total_problems_in_batch'):
            progress_str = f"[{self._current_problem_idx + 1}/{self._total_problems_in_batch}] "

        # Calculate merge statistics
        merge_pct = (tree.merge_count / tree.state_count * 100) if tree.state_count > 0 else 0

        print(f"{progress_str}Problem {problem.id}: {elapsed:.1f}s | "
              f"{success_count}/{len(trajs)} success | {len(samples)} samples | "
              f"{len(tree.nodes)} nodes ({tree.state_count} states, {tree.merge_count} merged = {merge_pct:.0f}%)",
              flush=True)
        return samples

    def build_batch(self, problems: list[Problem]) -> list[TrainingSample]:
        import time

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
                    print(f"  → Batch ETA: {eta:.0f}s ({avg_time:.1f}s/problem)", flush=True)

        all_samples = [s for g in groups for s in g]
        return all_samples

    # -- torch update (cluster) --------------------------------------------------
    def _seq_logprob(self, model, messages: list, continuation: str):
        """Log-prob of `continuation` under `model`, scored on the SAME chat-template
        rendering the policy generated with (fixes the prompt-format mismatch). The
        prompt span is found by longest-common-prefix of token ids, so tokenizer merges
        at the prompt/continuation seam can't misalign the masked target."""
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
        ).input_ids.to(model.device)
        p_len = _common_prefix_len(p_ids, f_ids[0].tolist())
        if f_ids.shape[1] <= p_len:  # nothing to score
            return torch.zeros((), device=model.device)
        logits = model(f_ids).logits[:, :-1, :]
        targets = f_ids[:, 1:]
        tok_logp = (
            torch.log_softmax(logits, dim=-1)
            .gather(-1, targets.unsqueeze(-1))
            .squeeze(-1)[0]
        )
        return tok_logp[p_len - 1 :].sum()

    def _update(self, samples: list[TrainingSample]) -> dict:
        import torch

        if not hasattr(self, "_opt"):
            params = [p for p in self.model.parameters() if p.requires_grad]
            self._opt = torch.optim.AdamW(params, lr=self.cfg.lr)
        # normalize advantages (whitening) for stability
        advs = [s.advantage for s in samples]
        mean = sum(advs) / len(advs)
        var = sum((a - mean) ** 2 for a in advs) / max(len(advs) - 1, 1)
        std = var**0.5 or 1.0

        self._opt.zero_grad()
        n = max(len(samples), 1)
        running = 0.0

        # Process samples in micro-batches to avoid OOM
        micro_batch_size = 8  # Process 8 samples at a time
        for i, s in enumerate(samples):
            adv = (s.advantage - mean) / std
            lp = self._seq_logprob(self.model, s.messages, s.continuation)
            loss_s = -(adv * s.weight) * lp
            if self.ref_model is not None:
                with torch.no_grad():
                    rlp = self._seq_logprob(self.ref_model, s.messages, s.continuation)
                loss_s = loss_s + self.cfg.kl_coef * (lp - rlp)
            # Per-sample backward so the autograd graph for each sample is freed
            # immediately. Accumulating `total = total + loss_s` across the whole batch
            # keeps every sample's graph alive at once -> OOM with group_size*depth*problems
            # sequences on one GPU. Dividing by n keeps the gradient == mean-loss gradient.
            (loss_s / n).backward()
            running += float(loss_s.detach())

            # Clear cache every micro_batch_size samples to free fragmented memory
            if (i + 1) % micro_batch_size == 0:
                torch.cuda.empty_cache()

        self._opt.step()
        torch.cuda.empty_cache()  # Clear cache after optimizer step
        return {"loss": running / n, "n_samples": len(samples)}

    def _save(self, output_dir: str, tag: str) -> None:
        path = os.path.join(output_dir, tag)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[grpo_tree] checkpoint saved -> {path}")

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
        bp = self.cfg.batch_problems
        step = 0
        total_steps = (len(problems) + bp - 1) // bp * self.cfg.epochs

        print(f"\n{'='*60}")
        print(f"Starting RL Training")
        print(f"{'='*60}")
        print(f"Total problems: {len(problems)} | Batch size: {bp} | Epochs: {self.cfg.epochs}")
        print(f"Total gradient steps: {total_steps}")
        print(f"{'='*60}\n", flush=True)

        for _ in range(self.cfg.epochs):
            for i in range(0, len(problems), bp):
                batch_end = min(i + bp, len(problems))
                batch_problems = problems[i:batch_end]

                print(f"\n>>> Step {step + 1}/{total_steps} | Problems {i + 1}-{batch_end}/{len(problems)}", flush=True)

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

                    print(f"  Rollouts: {t1 - t0:.1f}s | Update: {t2 - t1:.1f}s | "
                          f"Loss: {update_stats.get('loss', 0):.4f} | Samples: {len(samples)}", flush=True)
                    print(f"  ETA: {eta_minutes:.1f} min ({avg_step_time:.1f}s/step)\n", flush=True)

                    stats.append(update_stats)
                    step += 1
                    if save_every and output_dir and step % save_every == 0:
                        self._save(output_dir, f"checkpoint-{step}")
                else:
                    print(f"  WARNING: No samples generated!\n", flush=True)

        print(f"\n{'='*60}")
        print(f"Training Complete! {step} steps finished.")
        print(f"{'='*60}\n", flush=True)
        return stats


def _load_judge_model(rl_cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = rl_cfg.get("judge_model", "Qwen/Qwen2.5-7B-Instruct")
    kw = {"torch_dtype": "auto", "device_map": "auto"}
    if rl_cfg.get("judge_load_in_4bit"):
        from transformers import BitsAndBytesConfig

        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
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


def run_grpo_tree(model, tokenizer, rl_cfg: dict):
    """Entry from run.py. In-process policy (the trained model) + in-process judge."""
    _assert_single_process(rl_cfg["output_dir"])

    from .embedder import build_state_matcher

    policy = TransformersPolicy(
        model, tokenizer, temperature=rl_cfg.get("temperature", 0.8)
    )
    print(f"Loading judge model: {rl_cfg.get('judge_model', 'Qwen/Qwen2.5-7B-Instruct')}...", flush=True)
    jm, jt = _load_judge_model(rl_cfg)
    judge = TransformersJudge(jm, jt)
    matcher = build_state_matcher(rl_cfg.get("merge", {}))  # semantic merging by default

    problems = load_problems_jsonl(
        rl_cfg["problems_path"], limit=rl_cfg.get("max_problems")
    )

    cfg = TreeGRPOConfig(**rl_cfg.get("tree", {}))
    trainer = GRPOTreeTrainer(model, tokenizer, policy, judge, cfg=cfg, matcher=matcher)
    stats = trainer.train(
        problems,
        output_dir=rl_cfg["output_dir"],
        save_every=rl_cfg.get("save_every", 0),
    )
    model.save_pretrained(rl_cfg["output_dir"])
    tokenizer.save_pretrained(rl_cfg["output_dir"])
    print(f"\n[grpo_tree] Complete! {len(stats)} updates, final loss: {stats[-1]['loss']:.4f}" if stats else "\n[grpo_tree] Complete!")

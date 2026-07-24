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
        print(f"[grpo_tree._tree_samples] Starting tree build for problem {problem.id}...", flush=True)
        t0 = time.time()
        tree, trajs = self.engine.build_tree(problem, self.cfg.group_size, self.weights)
        t1 = time.time()
        print(f"[grpo_tree._tree_samples] Tree built in {t1 - t0:.2f}s for problem {problem.id}. "
              f"Got {len(trajs)} trajectories.", flush=True)

        print(f"[grpo_tree._tree_samples] Building samples from tree...", flush=True)
        samples = build_samples(
            tree, trajs, problem, self.weights, base_weight=self.cfg.base_weight
        )
        t2 = time.time()
        print(f"[grpo_tree._tree_samples] Built {len(samples)} samples in {t2 - t1:.2f}s. "
              f"Total time: {t2 - t0:.2f}s", flush=True)
        return samples

    def build_batch(self, problems: list[Problem]) -> list[TrainingSample]:
        print(f"[grpo_tree.build_batch] Building batch for {len(problems)} problem(s)...", flush=True)
        if self.cfg.num_workers and self.cfg.num_workers > 1:
            print(f"[grpo_tree.build_batch] Using {self.cfg.num_workers} workers.", flush=True)
            with ThreadPoolExecutor(max_workers=self.cfg.num_workers) as ex:
                groups = list(ex.map(self._tree_samples, problems))
        else:
            print(f"[grpo_tree.build_batch] Processing problems sequentially...", flush=True)
            groups = [self._tree_samples(p) for p in problems]

        all_samples = [s for g in groups for s in g]
        print(f"[grpo_tree.build_batch] Batch complete. Total samples: {len(all_samples)}", flush=True)
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
        for s in samples:
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
        self._opt.step()
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
        print("[grpo_tree.train] Setting models to eval mode...", flush=True)
        self.model.eval()
        if self.ref_model is not None:
            self.ref_model.eval()
        print("[grpo_tree.train] Models set to eval mode.", flush=True)

        stats = []
        bp = self.cfg.batch_problems
        step = 0
        print(f"[grpo_tree.train] Starting {self.cfg.epochs} epoch(s) with {len(problems)} problems, "
              f"batch_problems={bp}", flush=True)

        for epoch in range(self.cfg.epochs):
            print(f"[grpo_tree.train] === Epoch {epoch + 1}/{self.cfg.epochs} ===", flush=True)
            for i in range(0, len(problems), bp):
                batch_end = min(i + bp, len(problems))
                batch_problems = problems[i:batch_end]
                print(f"[grpo_tree.train] Step {step + 1}: Processing problems {i + 1}-{batch_end}/{len(problems)}...", flush=True)

                t0 = time.time()
                print(f"[grpo_tree.train]   Building batch (rollouts + samples)...", flush=True)
                samples = self.build_batch(batch_problems)
                t1 = time.time()
                print(f"[grpo_tree.train]   Batch built in {t1 - t0:.2f}s, got {len(samples)} samples.", flush=True)

                if samples:
                    print(f"[grpo_tree.train]   Running gradient update...", flush=True)
                    update_stats = self._update(samples)
                    t2 = time.time()
                    print(f"[grpo_tree.train]   Update completed in {t2 - t1:.2f}s. Loss: {update_stats.get('loss', 'N/A')}", flush=True)
                    stats.append(update_stats)
                    step += 1
                    if save_every and output_dir and step % save_every == 0:
                        print(f"[grpo_tree.train]   Saving checkpoint at step {step}...", flush=True)
                        self._save(output_dir, f"checkpoint-{step}")
                else:
                    print(f"[grpo_tree.train]   WARNING: No samples generated for this batch!", flush=True)

        print(f"[grpo_tree.train] Training loop completed. Total steps: {step}", flush=True)
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
    import time

    print("[grpo_tree] Starting run_grpo_tree...", flush=True)
    _assert_single_process(rl_cfg["output_dir"])

    from .embedder import build_state_matcher

    print("[grpo_tree] Creating TransformersPolicy...", flush=True)
    policy = TransformersPolicy(
        model, tokenizer, temperature=rl_cfg.get("temperature", 0.8)
    )
    print("[grpo_tree] Policy created successfully.", flush=True)

    print(f"[grpo_tree] Loading judge model: {rl_cfg.get('judge_model', 'Qwen/Qwen2.5-7B-Instruct')}...", flush=True)
    jm, jt = _load_judge_model(rl_cfg)
    print("[grpo_tree] Judge model loaded successfully.", flush=True)

    print("[grpo_tree] Creating TransformersJudge...", flush=True)
    judge = TransformersJudge(jm, jt)
    print("[grpo_tree] Judge created successfully.", flush=True)

    print("[grpo_tree] Building state matcher...", flush=True)
    matcher = build_state_matcher(rl_cfg.get("merge", {}))  # semantic merging by default
    print("[grpo_tree] State matcher built successfully.", flush=True)

    print(f"[grpo_tree] Loading problems from {rl_cfg['problems_path']}...", flush=True)
    problems = load_problems_jsonl(
        rl_cfg["problems_path"], limit=rl_cfg.get("max_problems")
    )
    print(f"[grpo_tree] Loaded {len(problems)} problems.", flush=True)

    cfg = TreeGRPOConfig(**rl_cfg.get("tree", {}))
    print(f"[grpo_tree] Config: group_size={cfg.group_size}, max_depth={cfg.max_depth}, "
          f"batch_problems={cfg.batch_problems}, epochs={cfg.epochs}", flush=True)

    print("[grpo_tree] Creating GRPOTreeTrainer...", flush=True)
    trainer = GRPOTreeTrainer(model, tokenizer, policy, judge, cfg=cfg, matcher=matcher)
    print("[grpo_tree] Trainer created successfully.", flush=True)

    print("[grpo_tree] Starting training...", flush=True)
    start_time = time.time()
    stats = trainer.train(
        problems,
        output_dir=rl_cfg["output_dir"],
        save_every=rl_cfg.get("save_every", 0),
    )
    elapsed = time.time() - start_time
    print(f"[grpo_tree] Training completed in {elapsed:.2f}s", flush=True)

    print("[grpo_tree] Saving final model...", flush=True)
    model.save_pretrained(rl_cfg["output_dir"])
    tokenizer.save_pretrained(rl_cfg["output_dir"])
    print(
        f"[grpo_tree] done. problems={len(problems)} updates={len(stats)} "
        f"last={stats[-1] if stats else None}"
    )

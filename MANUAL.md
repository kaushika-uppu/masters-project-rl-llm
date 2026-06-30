# Project Manual — RL for Critical Reasoning on Proofs

A reviewer's guide to everything that was built: what each piece does, **where to look in
the code**, how it fits together, how to run it, and the open issues / HPC gotchas to fix
before the first real run.

> Mental model in one sentence: **SFT** warms up a model on prove-or-disprove proofs, then
> **RL** has it reason step-by-step into a *tree*, a *judge* scores each step, and we reward
> correct-but-novel paths with credit concentrated on the *critical* decisions — then we
> test whether that reasoning **transfers** to RiddleBench.

---

## 1. Repo map — where to look

| Area | File | What it is / read this to understand… |
|---|---|---|
| **Data** | `src/data/deeptheorem.py` | Loading DeepTheorem, difficulty split, true/false variant parsing, `<step>` SFT formatting. The schema is documented at the top. |
| **SFT** | `src/training/sft.py` | `run_sft` + `get_deeptheorem` — how proofs become prove-or-disprove training examples. |
| **Entry** | `src/training/run.py` | Single entry point. Loads model, runs SFT and/or RL based on the YAML. |
| **RL dispatch** | `src/training/rl/run_rl.py` | Picks `grpo_baseline` vs `grpo_tree`. |
| **Contracts** | `src/training/rl/types.py` | `Problem`, `StepJudgement`, and the `Policy`/`Judge` protocols. **Start here** to understand the interfaces. |
| **Rollout engine** | `src/training/rl/rollout.py` | `RolloutEngine` — **lock-step batched** tree build: all active rollouts' next steps generate + are judged together per depth; in-context retry; trajectories. |
| **Tree / DAG** | `src/training/rl/tree.py` | `ProofTree`, `Node`, `ChildEdge` — node merging + Monte-Carlo value pooling. |
| **State key** | `src/training/rl/state.py` + `merge.py` | How two states are judged "the same" for merging (the crux dependency). |
| **Reward** | `src/training/rl/reward.py` | Endpoint verdict, step validity, redundancy penalty, correctness-gated novelty. |
| **Advantage** | `src/training/rl/advantage.py` | MC edge advantages + criticality (the "critical node" math). |
| **Samples** | `src/training/rl/sample_builder.py` | Turns a tree+trajectories into weighted `(prompt, step, advantage, weight)` training samples. |
| **Tree trainer** | `src/training/rl/grpo_tree.py` | `GRPOTreeTrainer` + `run_grpo_tree` — the policy-gradient update (the contribution). |
| **Baseline trainer** | `src/training/rl/baseline_grpo.py` | TRL GRPO, trajectory-level verdict reward (the de-risking run). |
| **In-process models** | `src/training/rl/inproc.py` | `TransformersPolicy` / `TransformersJudge` — generation with no servers; `*_batch` methods + `_generate_batch` (left-padded batched generation). |
| **(optional) served** | `src/training/rl/policy.py` + `src/judge/step_judge.py` | HTTP/vLLM-served Policy/Judge. **Not used by default.** |
| **Judge prompts** | `src/judge/prompts.py` | The judge's system prompt + state-summary contract. |
| **Problems input** | `src/training/rl/problems.py` | `load_problems_jsonl` — the curated examples the pipeline consumes. |
| **Eval (transfer)** | `evaluation/benchmarks/riddlebench.py` | (pre-existing) RiddleBench. |
| **Eval (in-domain)** | `evaluation/benchmarks/deeptheorem_eval.py` | Prove-or-disprove verdict accuracy from a provided jsonl. |
| **Configs** | `src/training/configs/{simple,rl_baseline,rl_tree}.yaml` | The three runs. |
| **Tests** | `tests/test_deeptheorem_data.py`, `test_rl_engine.py`, `test_rl_samples.py` | What's verified in-sandbox (no GPU). Run these first. |

---

## 2. End-to-end data flow

```
DeepTheorem rows ──(curation, EXTERNAL)──▶ data/rl_problems.jsonl  +  data/deeptheorem_eval.jsonl
        │
        ▼  src/data/deeptheorem.py (SFT only)
   SFT examples  ──▶  src/training/sft.py  ──▶  checkpoints/simple_deeptheorem   (= SFT baseline + RL init)
        │
        ▼  RL: src/training/run.py → run_rl.py
   ┌────────────────────────── grpo_tree (the contribution) ──────────────────────────┐
   │ load_problems_jsonl → for each Problem:                                            │
   │   RolloutEngine (rollout.py): policy proposes <step> → judge validates →           │
   │      retry-on-fail → ProofTree (tree.py) merges states (merge.py) →                │
   │      trajectories with terminal reward                                             │
   │   advantage.py: Monte-Carlo node values → per-step advantages + criticality        │
   │   sample_builder.py: (prompt, step, advantage, weight) samples                     │
   │   grpo_tree.py _update: −(adv·weight)·logprob(step) [+ KL]  → AdamW step            │
   └────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼  checkpoints/rl_tree
   Eval: scripts/evaluate.py --benchmarks riddlebench deeptheorem
```

---

## 3. Subsystem walkthroughs (with pointers)

### 3.1 Data → prove-or-disprove
`src/data/deeptheorem.py`. The dataset's `pos`/`neg` columns are matched true/false
variants, each a dict `{question, response, truth_value}`. `parse_variants()` reads the
per-variant `truth_value` as the label (no inference). `build_sft_examples()` formats each
variant as a chat example ending in `Verdict: PROVED/DISPROVED`, with the proof split into
`<step>` blocks. Difficulty `<=7.0` → SFT pool; `>7.0` → RL pool (`split_by_difficulty`).
**Note:** curation/subsetting is external — the RL/eval pipelines read jsonl files, not the
HF dataset.

### 3.2 SFT
`src/training/sft.py` → `get_deeptheorem`. Default `task_format="prove_or_disprove"` so the
SFT model speaks the exact format RL expects (no train→RL drift). `max_length=2048`. Output
checkpoint is both your **SFT baseline** and the **RL initialization**.

### 3.3 The Policy and Judge (interfaces)
`src/training/rl/types.py` defines two protocols:
- `Policy.propose_steps(problem, history, k)` → k candidate next steps; `revise_step(...)` for retries.
- `Judge.judge_step(problem, history, step)` → `StepJudgement(valid, reason, state_summary, is_terminal, verdict)`.

Default implementations are **in-process** (`inproc.py`): the policy is *the model being
trained* (on-policy), the judge is a second model (`judge_model`, optional 4-bit). Swapping
in a fine-tuned judge later = change one config line.

### 3.4 Rollout → tree → merge  (lock-step batched)
`rollout.py::RolloutEngine.build_tree` runs `group_size` independent rollouts **in
lock-step by depth**: at each depth it gathers every active rollout's history, generates
their next steps in ONE batched call (`_propose`), and judges them in ONE batched call
(`_judge`); invalid steps are recorded as dead-end leaves and **retried in-context**
(batched, bounded by `retry_budget`). This is the throughput fix — one model forward per
depth (batch = #active rollouts) instead of one per rollout per depth. Batching is
opportunistic: `_propose/_revise/_judge` use the policy/judge `*_batch` methods if present
(`inproc.py`) and otherwise fall back to per-item calls (served clients, stubs). Each
trajectory's terminal reward is pooled into every node on its path
(`tree.py::record_rollout`) → **Monte-Carlo node values, no value network**. Merging
(`merge.py`) keys on the judge's `state_summary` (`state.py::canonicalize`), conservative
(under-merge). Siblings stay independent — each rollout conditions only on its own history.

### 3.5 Reward
`reward.py`: `endpoint_reward` (verdict vs label — the verifiable anchor), `step_validity_reward`,
`redundancy_penalty` (loops), `novelty_bonus` (**gated on `success`** — zero if the path is
wrong). Weights in `RewardWeights`.

### 3.6 Advantage + criticality
`advantage.py`: `edge_advantages` = V(child) − V(parent); `criticality(node)` = spread of
child values × √visits → the decision forks. `sample_builder.build_samples` combines these
into per-step `advantage` and `weight = base + criticality(parent)`.

### 3.7 The update
`grpo_tree.py::GRPOTreeTrainer._update`: whitens advantages, then per-sample
`−(adv·weight)·logprob(step)` (+ optional KL vs `ref_model`), **per-sample `backward()`** to
bound memory, then one AdamW step. `train()` checkpoints every `save_every` updates.
`build_batch` is pure and unit-tested; `_update` needs GPU.

### 3.8 Baseline
`baseline_grpo.py`: TRL `GRPOTrainer`, reward = parsed verdict vs label, optional `use_vllm`.
This sidesteps all the custom-update risk — **run it first.**

### 3.9 Eval
`deeptheorem_eval.py` reads `DEEPTHEOREM_EVAL_PATH` (default `data/deeptheorem_eval.jsonl`),
prompts prove-or-disprove, parses the verdict, scores vs label. RiddleBench is pre-existing.
Both run through `scripts/evaluate.py` (SLURM via `scripts/generate_slurm_script.py`).

---

## 4. How to run

```bash
# 0. tests (no GPU)
python tests/test_deeptheorem_data.py && python tests/test_rl_engine.py && python tests/test_rl_samples.py

# 1. SFT (also produces the SFT baseline checkpoint)
python -m src.training.run src/training/configs/simple.yaml

# 2a. Baseline RL  (needs data/rl_problems.jsonl)
python -m src.training.run src/training/configs/rl_baseline.yaml
# 2b. Tree RL
python -m src.training.run src/training/configs/rl_tree.yaml

# 3. Eval (needs data/deeptheorem_eval.jsonl for the in-domain one)
python scripts/evaluate.py --inference-fn <fn> --benchmarks riddlebench deeptheorem
```

**You must provide two files** (curation, external): `data/rl_problems.jsonl`
(`{id, statement, label}`) and `data/deeptheorem_eval.jsonl` (same shape). Until then the RL
and in-domain eval runs raise a clear `FileNotFoundError`.

---

## 5. Code-review findings (ranked)

**A. Prompt-format mismatch in the tree trainer — FIXED.** A single source of truth,
`policy.build_step_messages(problem, history)`, now builds the per-step chat prompt for BOTH
generation (`inproc.TransformersPolicy` / `VLLMPolicy`) and scoring. `sample_builder` stores
that `messages` list plus the `continuation` (`reconstruct_continuation(step)` — re-wraps a
normal step as `<step>…</step>`, leaves a `Verdict:` line bare). `grpo_tree._seq_logprob`
renders the prompt with `apply_chat_template(..., add_generation_prompt=True)` and scores the
continuation, so the log-prob matches what the policy actually generated.

**B. Token-boundary alignment — FIXED.** `_seq_logprob` now finds the prompt span by
`_common_prefix_len` of the token ids of `prompt` vs `prompt+continuation` (both tokenized
with `add_special_tokens=False`), so a tokenizer merge at the seam can't misalign the masked
target.

**C. Dropout during forward — FIXED.** `GRPOTreeTrainer.train()` calls `self.model.eval()`
(and `ref_model.eval()`), disabling LoRA/attention dropout so generation and the PG log-probs
are deterministic given the sampled tokens (autograd still flows in eval mode).

**D. Judge JSON reliability (the practical bottleneck).** `TransformersJudge` expects strict
JSON from an off-the-shelf instruct model with no grammar constraint. Expect frequent parse
failures → steps wrongly marked invalid → rollouts die early / noisy reward. Mitigation:
few-shot the judge, or use constrained/JSON decoding.

**E. Merge depends on consistent `state_summary`.** Merging/loops/MC-pooling all key on the
judge's free-text state summary. If it phrases the same state differently each time, nothing
merges → single-visit nodes → criticality ≈ 0 → advantages degenerate. This is the make-or-
break dependency; watch the tree size vs rollouts and consider a structured state format.

**F. Cold-start sparsity.** `novelty` and `endpoint_reward` are gated on a *correct terminal
verdict*; early on, many rollouts hit `max_depth` without terminating (verdict `None` →
reward 0, success False). All-zero groups give no gradient. Mitigations: start on easier
problems, raise `max_depth`, or give partial credit for reaching a terminal at all.

**G. Simplifications (acceptable for v1, note for the writeup).** Pure REINFORCE with
advantage whitening — no PPO clipping / importance ratio; KL only if you pass a `ref_model`
(default `None` → no anchor against drift). The PG forward (`_update`) is still batch-1 per
sample (correct, memory-safe, but slow — a later optimization if needed).

**Throughput — rollout generation is now batched (was the main bottleneck).** `build_tree`
runs the GRPO group in lock-step: one batched generation + one batched judge call per depth
(`inproc._generate_batch`, left-padded), instead of `group_size × depth` serial calls. This
is the fix for the ~1000-question scale. Remaining throughput levers if still too slow:
vLLM-library generation (needs weight-sync) and batching the `_update` forward (G).

Nothing is a syntax/import bug — all 21 sandbox tests pass and every module imports.

---

## 6. HPC (SJSU) specifics & gotchas

- **No training SLURM script exists.** `scripts/generate_slurm_script.py` builds sbatch
  scripts for **eval only** (it wraps `scripts/evaluate.py`). You'll need a separate sbatch
  for `python -m src.training.run ...`. Reuse that script's preamble: `source v_env/bin/activate`,
  `export PYTHONPATH="$PWD:$PYTHONPATH"`, the `TMPDIR` scratch trap, partition `gpuqm`.
- **Single GPU, single process for `grpo_tree`.** `_assert_single_process` aborts if launched
  with `SLURM_NTASKS>1`/`torchrun`. Request **one task, one GPU** (`--gres=gpu:1`,
  `--ntasks=1`). The config pins the policy with `device_map: {"": 0}`; SLURM's
  `CUDA_VISIBLE_DEVICES` makes index 0 the assigned GPU. Multi-GPU is only for `grpo_baseline`
  (via `accelerate launch`, and set `device_map: null` there).
- **Memory.** One GPU holds policy 7B (bf16, ~15 GB) + LoRA + judge 7B (4-bit, ~5–6 GB) +
  generation KV cache + the PG forward. Fine on H100/A100-80GB; tight on 40–48 GB (L40S/A100-40)
  — drop `group_size`/`max_depth` or keep the judge 4-bit. The eval sbatch's `--mem=32G`
  (system RAM) is low for loading two 7B models; bump for training.
- **Wall-time / preemption.** `save_every` writes `checkpoint-N/`; resume by pointing
  `from_checkpoint` at the latest. Set `save_every` sensibly vs the queue's time limit.
- **`use_vllm` colocation (baseline).** TRL's vLLM generation shares the GPU with training;
  `vllm_gpu_memory_utilization: 0.3` is set, but on one GPU this is fragile and version-
  sensitive. If it OOMs or errors, set `use_vllm: false` first to confirm the run works.
- **TRL version drift.** `GRPOConfig` arg names (`num_generations`, `use_vllm`, etc.) move
  between TRL releases — pin a TRL version and adjust `baseline_grpo.py` if it errors.
- **HF download cache.** First run pulls Qwen + the judge from HuggingFace; set `HF_HOME` to
  a scratch/project dir and pre-download on a login node if compute nodes are offline.

---

## 7. Remaining gaps (before a real run)

1. **Provide the data files** — `data/rl_problems.jsonl` and `data/deeptheorem_eval.jsonl`
   (curation, with the eval set disjoint from training).
2. ~~Fix trainer findings A–C~~ **DONE** (see §5). Still validate on GPU: confirm the
   `_common_prefix_len` split lands correctly for your tokenizer (a quick assert that the
   scored token count > 0 and equals the continuation length).
3. **An inference function for eval** — `scripts/evaluate.py` needs a registered
   `--inference-fn` that loads your SFT/RL checkpoint and generates (check `src/function_registry.py`
   / `src/inference/`); the new `deeptheorem` benchmark supplies the prompt but not the model.
4. **Judge hardening** — few-shot or constrained JSON; and an offline judge-accuracy check
   (formal datasets with injected errors) to report a false-positive rate.
5. **A training sbatch** for `run.py` (see §6).
6. **Logging** — no metrics sink wired (`report_to: none`); add wandb/tensorboard to see
   reward/entropy/tree-size during runs.
7. **Merge/state validation** — sanity-check that real judge summaries actually merge
   (finding E) on a handful of problems before a long run.

---

## 8. What's verified vs not

- **Verified in-sandbox (no GPU):** data parsing, **lock-step batched** tree build + merge
  (both the batched path and the per-item fallback), retry/self-correction, reward
  components, MC advantages + criticality, sample building (incl. chat `messages` +
  `continuation`), parallel `build_batch`, jsonl loading. (`tests/`, 21 tests.)
- **Not yet exercised (needs GPU/cluster):** the torch `_update`, real in-process batched
  generation (`_generate_batch` left-padding), the judge model, TRL baseline, end-to-end
  SFT→RL, and eval.

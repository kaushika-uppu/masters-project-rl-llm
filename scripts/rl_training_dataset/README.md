# RL training dataset builder (DeepTheorem, LIMO-style filtering)

Builds a ~1000-1500 question DeepTheorem RL training set by rolling out
Qwen2.5-7B-Instruct 8 times per theorem (4 on the `pos` variant, 4 on the
`neg` variant) and dropping anything the model got right every time (too
easy) or wrong every time (too hard/unlearnable) -- the same idea as
`scripts/filter_subsets.py` + `scripts/deeptheorem_phase1/2_grader.py`,
just consolidated into three explicit stages so it's easy to re-run any one
of them independently:

```
scripts/rl_training_dataset/
  prepare_candidates.py   # 1. exclude SFT+holdout sets, stratified-sample, expand pos/neg
  generate_rollouts.py    # 2. vLLM rollouts, 4 per variant row, SLURM-chunk friendly
  grade_and_filter.py     # 3. grade, purge 0/8 and 8/8, trim to target, write final set
```

## Quick start

```bash
# 1. pick candidates (excludes SFT + holdout sets; see "Exclusion sets" below)
python scripts/rl_training_dataset/prepare_candidates.py \
    --n-candidates 3000

# 2. rollouts (run per-chunk, e.g. as a SLURM array job -- see below)
python scripts/rl_training_dataset/generate_rollouts.py \
    --chunk-id 0 --chunk-size 500

# 3. grade + filter to the final training set
python scripts/rl_training_dataset/grade_and_filter.py
```

Output: `scripts/rl_training_dataset/final_rl_training_set.jsonl`, in the
schema `src/training/rl/problems.py::load_problems_jsonl` already expects
(`id`, `statement`, `label`, plus `domain`/`difficulty`/`reference`) -- no
further reformatting needed to plug it into the existing RL pipeline.

A `purged_theorems.jsonl` audit file is also written, listing everything
dropped for being always-right or always-wrong, with its `correct_count`,
so you can sanity-check the purge before training.

## Exclusion sets ("holdout eval set" and "SFT training set")

Neither file exists in this checkout (`data/` is gitignored everywhere, and
the repo has produced these under different names on different branches).
Both are **optional** -- if a path doesn't exist, `prepare_candidates.py`
prints a warning and just skips that exclusion, per your instructions.

| Set | Default path (`--sft-exclude` / `--holdout-exclude`) | Where that path comes from |
|---|---|---|
| SFT training set | `data/sft_reference_training.jsonl` | `src/training/configs/sft_v2.yaml` on `miranda/sft_v2` |
| Holdout eval set | `data/dt_eval_dataset.jsonl` | `scripts/create_dt_eval.py` on `main` |

If your real files live somewhere else, just pass `--sft-exclude` /
`--holdout-exclude` with the actual path.

**Why matching is by normalized statement text, not `id`:** the repo's own
scripts disagree on what "id" means for a DeepTheorem row -- sometimes it's
the HF dataset's real `id`, sometimes `int(row.name)` (a positional index
into whatever dataframe happened to be loaded), sometimes a derived
`{id}_pos`/`{id}_neg` string. Rather than assume one of those, exclusion
matches on whitespace/case-folded statement text (`ori_question`,
`informal_theorem`, `pos.question`, `neg.question`, etc. -- whichever fields
are present), the same trick `scripts/create_sft_eval.py` already uses to
reconcile a subset file back to the source HF dataset. A theorem is
excluded if *any* of its own texts (base question, pos, neg) shows up in
either exclusion file, so you're protected against test/train leakage in
either direction even if the file's schema doesn't match exactly what's
assumed.

## Sizing the initial candidate pool

`--n-candidates` (default 3000 theorems -> 6000 variant rows, since each
theorem expands into a pos + neg row) is the "how many we run rollouts on"
knob -- your call, per the brief. The default assumes a fairly aggressive
LIMO-style purge (40-70% of theorems dropped as too-easy/too-hard), which
comfortably clears the 1000-1500 target even in the worst case. `grade_and_
filter.py` tells you exactly how many survived and:
- auto-trims down (stratified by domain x difficulty) if you're over 1500,
- warns loudly if you're under 1000, so you know to bump `--n-candidates`
  and generate more rollouts.

Stratification (both at candidate-selection time and at final-trim time) is
by `domain` (first entry if it's a list) x difficulty quantile bucket, to
keep the final set representative of the full DeepTheorem distribution
rather than skewed toward whatever happened to survive.

## Prompt / verdict format

Uses the `<step>...</step>` + `\boxed{proved|disproved}` convention from
the "fix prompt inconsistency" commit on `miranda/sft_v2`
(`evaluation/benchmarks/deeptheorem_eval.py`), per your call. Grading
(`grade_and_filter.py::extract_verdict`) is ported directly from that same
commit and accepts both `\boxed{proved|disproved}` and the older
`Verdict: PROVED/DISPROVED` line, taking the last match in the text -- so
old-format rollouts (if you ever mix in older data) still grade correctly.

## Model

`Qwen/Qwen2.5-7B-Instruct` (`--model`), matching `src/training/configs/
sft.yaml` / `sft_v2.yaml`. Note this differs from `scripts/
deeptheorem_limo_phase1.py`, which uses `Qwen/Qwen2.5-Math-7B-Instruct` --
worth double-checking that's intentional for your setup, since the two are
different checkpoints.

## Running on the cluster (SLURM)

`generate_rollouts.py` is chunked (`--chunk-id`/`--chunk-size`) the same way
as `scripts/deeptheorem_phase1.py`/`deeptheorem_phase2.py`, so it drops into
a SLURM array job the same way, e.g. modeled on `scripts/hpc_scripts/
limo_deeptheorem.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=rl_dataset_rollouts
#SBATCH --output=logs/rl_dataset_rollouts/%A_%a.out
#SBATCH --error=logs/rl_dataset_rollouts/%A_%a.err
#SBATCH --partition=gpuqm
#SBATCH --gres=gpu:1
#SBATCH --array=0-11        # e.g. 6000 rows / 500 per chunk = 12 chunks
#SBATCH --time=04:00:00

source v_env/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

python scripts/rl_training_dataset/generate_rollouts.py \
    --chunk-id $SLURM_ARRAY_TASK_ID --chunk-size 500
```

Then run `grade_and_filter.py` once all chunks finish (it reads every
`*.json` file in `--rollouts-dir`).

## A note on the git side of this

This branch (`miranda/rl_training_dataset`) was built and pushed from a
clean clone rather than your connected `rl` folder's checkout, because that
folder's mount into this sandbox blocks file *deletion* (not creation) --
a `git pull` mid-flight left a stray `.git/index.lock` and some leftover
temp objects that this sandbox can't clean up, though your Mac can, trivially:

```bash
cd /path/to/masters-project-rl-llm
rm -f .git/index.lock .git/objects/*/tmp_obj_*
git status   # should be clean; if any of the 3 files below still show as
             # modified, they're already fixed to match origin/main, so
             # `git checkout -- <file>` is safe
git fetch origin
git checkout miranda/rl_training_dataset
```

The three script files above are also copied straight into your `rl`
folder's checkout (at the same `scripts/rl_training_dataset/` path) so you
can look at them immediately without touching git.

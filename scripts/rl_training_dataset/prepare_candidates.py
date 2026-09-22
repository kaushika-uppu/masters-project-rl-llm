"""Phase 1 of the RL training-set build: pick a stratified candidate pool of
DeepTheorem theorems, held disjoint from the SFT training set and the eval
holdout set, and expand each into its pos/neg "prove or disprove" variants.

This mirrors the LIMO-filtering pattern already used for DeepTheorem
(scripts/deeptheorem_limo_phase1.py -> scripts/deeptheorem_phase1_grader.py
-> scripts/deeptheorem_phase2.py -> scripts/deeptheorem_phase2_grader.py ->
scripts/filter_subsets.py), just consolidated into three explicit stages:

    1. prepare_candidates.py   (this file)   -- pick + expand candidates
    2. generate_rollouts.py                  -- vLLM rollouts (4 pos + 4 neg)
    3. grade_and_filter.py                   -- grade, purge, trim to target

WHY EXCLUSION IS TEXT-BASED, NOT ID-BASED
------------------------------------------
Different scripts in this repo assign "id" differently for DeepTheorem rows:
  - the raw HF dataset's own `id` column,
  - `int(row.name)` (a positional index into whatever dataframe was loaded --
    see scripts/deeptheorem_phase1.py / dt_sft_phase1.py), or
  - a derived `{base_id}_pos` / `{base_id}_neg` string (see
    scripts/create_dt_eval.py / create_sft_eval.py).
These are NOT guaranteed to line up across files. To stay safe against that
drift, exclusion is matched on normalized statement text (ori_question /
informal_theorem / pos.question / neg.question, whitespace+case folded) --
the same trick create_sft_eval.py already uses to reconcile a subset file
back to the source HF dataset. A theorem is excluded if ANY of its own texts
(base question, pos variant, neg variant) appears in either exclusion set.

EXPECTED EXCLUSION FILE PATHS (per branch, override with flags if different)
------------------------------------------------------------------------------
  --sft-exclude      data/sft_reference_training.jsonl   (miranda/sft_v2)
  --holdout-exclude  data/dt_eval_dataset.jsonl           (main, via
                                                            create_dt_eval.py)
Both are optional. If a file is missing, this script prints a warning and
just doesn't exclude anything from it -- per instructions, don't block on
files that may not exist in this checkout (data/ is gitignored everywhere).
Each file's *schema* is sniffed at load time (see `iter_exclusion_texts`)
rather than assumed, since we don't have real copies to test against.

CHOOSING --n-candidates ("your call" on how many to run rollouts on)
------------------------------------------------------------------------------
The goal is ~1000-1500 SURVIVING theorem-variants after grade_and_filter.py
drops anything the model got right in all 8/8 rollouts (too easy) or wrong in
0/8 (too hard/unlearnable). LIMO-style purges on similar setups typically
remove somewhere between 40-70% of examples this way, and this pool is at
the variant level (2 rows per theorem), so as a starting point:

    default = 3000 theorems -> 6000 variant rows -> ~1800-3600 survivors
    (comfortably covers the 1000-1500 target even at a 70% purge rate)

This is a rough calibration, not a guarantee -- grade_and_filter.py will tell
you exactly what you got and warns loudly if you land outside [1000, 1500],
so it's easy to re-run this step with a bigger/smaller --n-candidates once
you've seen real pass-rate numbers from one batch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.deeptheorem import DeepTheoremColumns, parse_variants  # noqa: E402

DEFAULT_HF_NAME = "Jiahao004/DeepTheorem"


def normalize_text(text) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", "", str(text)).lower()


# --------------------------------------------------------------------------------------
# Exclusion set loading (schema-tolerant)
# --------------------------------------------------------------------------------------
def iter_exclusion_texts(row: dict):
    """Yield every candidate 'statement-like' text found in one jsonl row, for any of
    the schemas this pipeline's various scripts have produced historically:
      - per-theorem rows with ori_question / informal_theorem (+ optional pos/neg dicts)
      - per-variant rows with a flat 'statement' or 'tested_variant' field
    """
    for key in ("ori_question", "informal_theorem", "statement", "tested_variant", "question"):
        if row.get(key):
            yield row[key]

    for variant_key in ("pos", "neg"):
        v = row.get(variant_key)
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                v = None
        if isinstance(v, dict) and v.get("question"):
            yield v["question"]


def load_exclusion_set(path: str, label: str) -> set:
    if not path:
        return set()
    if not os.path.exists(path):
        print(f"[prepare_candidates] WARNING: {label} exclusion file not found at "
              f"'{path}'. Skipping this exclusion (nothing will be excluded from it).")
        return set()

    texts = set()
    n_rows = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_rows += 1
            for t in iter_exclusion_texts(row):
                norm = normalize_text(t)
                if norm:
                    texts.add(norm)

    print(f"[prepare_candidates] Loaded {label} exclusion set: {n_rows} rows -> "
          f"{len(texts)} unique normalized statement texts from '{path}'.")
    return texts


def theorem_texts(row: dict, cols: DeepTheoremColumns) -> list[str]:
    """All texts belonging to one HF row that must be checked against exclusion sets."""
    texts = []
    base = row.get(cols.statement) or row.get("ori_question")
    if base:
        texts.append(base)
    for v in parse_variants(row, cols, include_original=False, warn=False):
        texts.append(v.statement)
    return texts


# --------------------------------------------------------------------------------------
# Stratified sampling
# --------------------------------------------------------------------------------------
def domain_key(value) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else "unknown"
    return str(value) if value else "unknown"


def assign_difficulty_bucket(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    df = df.copy()

    def qbin(s: pd.Series):
        bins = min(n_bins, s.nunique())
        if bins < 2:
            return pd.Series(0, index=s.index)
        return pd.qcut(s, q=bins, labels=False, duplicates="drop")

    df["difficulty_bucket"] = qbin(df["difficulty"].astype(float))
    return df


def stratified_sample(df: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    if n_target >= len(df):
        return df

    rng = np.random.default_rng(seed)
    groups = list(df.groupby(["domain_key", "difficulty_bucket"]))
    total = len(df)

    picks = []
    remaining = n_target
    for i, (_, grp) in enumerate(groups):
        is_last = i == len(groups) - 1
        if is_last:
            n = remaining
        else:
            n = int(round(n_target * len(grp) / total))
        n = max(0, min(n, len(grp), remaining))
        if n > 0:
            chosen = rng.choice(grp.index.to_numpy(), size=n, replace=False)
            picks.append(df.loc[chosen])
            remaining -= n
    sampled = pd.concat(picks) if picks else df.iloc[0:0]

    # if rounding left us short, top up randomly from whatever's left
    if len(sampled) < n_target:
        leftover = df.drop(sampled.index)
        top_up = min(n_target - len(sampled), len(leftover))
        if top_up > 0:
            chosen = rng.choice(leftover.index.to_numpy(), size=top_up, replace=False)
            sampled = pd.concat([sampled, df.loc[chosen]])

    return sampled.sort_index()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-name", type=str, default=DEFAULT_HF_NAME)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--sft-exclude", type=str, default="data/sft_reference_training.jsonl",
                   help="Path to the SFT training set to exclude (schema-tolerant; skipped if missing).")
    p.add_argument("--holdout-exclude", type=str, default="data/dt_eval_dataset.jsonl",
                   help="Path to the holdout eval set to exclude (schema-tolerant; skipped if missing).")
    p.add_argument("--n-candidates", type=int, default=3000,
                   help="Number of theorems to sample BEFORE rollouts (each yields 2 variant rows). Default 3000 (see module docstring for the math).")
    p.add_argument("--n-bins", type=int, default=5, help="Difficulty quantile buckets per domain for stratification.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="scripts/rl_training_dataset/candidates.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    cols = DeepTheoremColumns()

    sft_excluded = load_exclusion_set(args.sft_exclude, "SFT training set")
    holdout_excluded = load_exclusion_set(args.holdout_exclude, "holdout eval set")
    excluded_texts = sft_excluded | holdout_excluded

    print(f"[prepare_candidates] Loading {args.hf_name} split={args.split} ...")
    from datasets import load_dataset
    ds = load_dataset(args.hf_name, split=args.split)

    rows = []
    n_excluded = 0
    for row in ds:
        texts = [normalize_text(t) for t in theorem_texts(row, cols)]
        if excluded_texts and any(t in excluded_texts for t in texts if t):
            n_excluded += 1
            continue
        difficulty = row.get(cols.difficulty)
        if difficulty is None:
            continue
        rows.append({
            "base_id": row.get("id"),
            "domain_key": domain_key(row.get(cols.domain)),
            "difficulty": float(difficulty),
            "_row": row,
        })

    print(f"[prepare_candidates] Excluded {n_excluded} theorems present in SFT/holdout sets.")
    pool_df = pd.DataFrame(rows)
    if pool_df.empty:
        raise RuntimeError("No candidate theorems left after exclusion -- check your exclusion files/schema.")

    pool_df = assign_difficulty_bucket(pool_df, args.n_bins)
    print(f"[prepare_candidates] Candidate pool after exclusion: {len(pool_df)} theorems.")

    sampled_df = stratified_sample(pool_df, args.n_candidates, args.seed)
    print(f"[prepare_candidates] Stratified sample: {len(sampled_df)} theorems "
          f"(requested {args.n_candidates}).")

    print("\nDomain x difficulty-bucket distribution (pool -> sampled):")
    for (dom, bucket), grp in pool_df.groupby(["domain_key", "difficulty_bucket"]):
        s = sampled_df[(sampled_df["domain_key"] == dom) & (sampled_df["difficulty_bucket"] == bucket)]
        print(f"  domain={dom!r:30s} bucket={int(bucket)}  pool={len(grp):5d}  sampled={len(s):5d}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    n_variants = 0
    n_theorems_no_variants = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for _, r in sampled_df.iterrows():
            variants = parse_variants(r["_row"], cols, include_original=False, warn=False)
            if not variants:
                n_theorems_no_variants += 1
                continue
            for v in variants:
                out = {
                    "id": f"{r['base_id']}_{v.meta.get('variant', 'unk')}",
                    "base_id": r["base_id"],
                    "variant": v.meta.get("variant"),
                    "statement": v.statement,
                    "label": v.label,
                    "reference": v.reference,
                    "domain": r["domain_key"],
                    "difficulty": r["difficulty"],
                }
                f.write(json.dumps(out) + "\n")
                n_variants += 1

    if n_theorems_no_variants:
        warnings.warn(f"{n_theorems_no_variants} sampled theorems had no parseable pos/neg "
                       f"variants and were skipped -- check the pos/neg columns for those rows.")

    print(f"\n[prepare_candidates] Wrote {n_variants} candidate variant rows "
          f"({n_variants // 2 if n_variants else 0} theorems x ~2 variants) to {args.output}")
    print("Next: scripts/rl_training_dataset/generate_rollouts.py --input "
          f"{args.output} --chunk-id 0 --chunk-size <N>")


if __name__ == "__main__":
    main()

"""Phase 3 of the RL training-set build: grade every rollout, aggregate each
theorem's pos+neg rollouts into one correct_count out of 8, purge anything
the model got right every time (too easy) or wrong every time (too hard /
unlearnable), and trim what's left to the ~1000-1500 target.

This is the LIMO-style purge, generalized from scripts/filter_subsets.py
(which already does exactly "keep min <= correct_count <= max") plus
scripts/deeptheorem_phase2_grader.py's scoring loop -- consolidated here so
it operates at the THEOREM level (pos + neg combined = 8 rollouts) rather
than the variant level, since "every question it gets right/wrong every
time" is about the theorem, not a single pos/neg row in isolation.

VERDICT EXTRACTION
--------------------
Ported from evaluation/benchmarks/deeptheorem_eval.py on miranda/sft_v2
(the "fix prompt inconsistency" commit). Accepts BOTH:
  - "Verdict: PROVED" / "Verdict: DISPROVED"   (older convention)
  - "\\boxed{proved}" / "\\boxed{disproved}"    (current convention, what
    generate_rollouts.py's prompt asks for)
and takes the LAST match in the text, matching the existing eval code.

OUTPUT
--------
Final dataset is written in the schema src/training/rl/problems.py's
`load_problems_jsonl` already expects: one JSON object per line with
{"id", "statement", "label", "domain", "difficulty", "reference"} -- so it
plugs directly into the existing RL pipeline with no further reformatting.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

_VERDICT_PATTERNS = (
    re.compile(r"verdict\s*:?\s*(proved|disproved)", re.IGNORECASE),
    re.compile(
        r"\\boxed\s*\{\s*(?:\\(?:text|mathrm)\s*\{\s*)?"
        r"(proved|disproved)\s*(?:\}\s*)?\}",
        re.IGNORECASE,
    ),
)


def extract_verdict(text: str):
    """Return the LAST explicit verdict ('proved'/'disproved') in either supported format."""
    matches = [
        (match.start(), match.group(1).lower())
        for pattern in _VERDICT_PATTERNS
        for match in pattern.finditer(text or "")
    ]
    return max(matches, key=lambda item: item[0])[1] if matches else None


def grade_rollout(rollout_text: str, label: bool) -> bool:
    verdict = extract_verdict(rollout_text)
    if verdict is None:
        return False  # ungraded/malformed output counts as incorrect, same as the existing graders
    predicted = verdict == "proved"
    return predicted == label


def load_rollout_rows(rollouts_dir: str) -> list[dict]:
    files = sorted(glob.glob(os.path.join(rollouts_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No rollout chunk files found in '{rollouts_dir}'. "
                                 "Run generate_rollouts.py first.")
    rows = []
    for fp in files:
        with open(fp) as f:
            rows.extend(json.load(f))
    print(f"[grade_and_filter] Loaded {len(rows)} variant rows from {len(files)} chunk file(s).")
    return rows


def aggregate_by_theorem(rows: list[dict]) -> pd.DataFrame:
    agg = defaultdict(lambda: {"correct": 0, "total": 0, "variants_seen": set(), "meta": None})
    for row in rows:
        base_id = row["base_id"]
        label = bool(row["label"])
        rollouts = row.get("rollouts", [])
        correct = sum(1 for r in rollouts if grade_rollout(r, label))

        entry = agg[base_id]
        entry["correct"] += correct
        entry["total"] += len(rollouts)
        entry["variants_seen"].add(row.get("variant"))
        if entry["meta"] is None:
            entry["meta"] = {
                "domain": row.get("domain"),
                "difficulty": row.get("difficulty"),
            }

    records = []
    for base_id, entry in agg.items():
        records.append({
            "base_id": base_id,
            "correct_count": entry["correct"],
            "total_rollouts": entry["total"],
            "n_variants": len(entry["variants_seen"]),
            "domain": entry["meta"]["domain"],
            "difficulty": entry["meta"]["difficulty"],
        })
    return pd.DataFrame(records)


def print_distribution(df: pd.DataFrame, col: str, title: str):
    print(f"\n{title}")
    print("-" * 65)
    max_total = int(df["total_rollouts"].max()) if len(df) else 0
    for n in range(max_total + 1):
        count = int((df[col] == n).sum())
        pct = 100 * count / len(df) if len(df) else 0
        bar = "#" * int(pct / 2)
        print(f"{n:>2}/{max_total} correct | {count:6d} | {bar} ({pct:.1f}%)")
    print("-" * 65)


def stratified_trim(df: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    if len(df) <= n_target:
        return df
    rng = np.random.default_rng(seed)

    def qbin(s: pd.Series, n_bins=5):
        bins = min(n_bins, s.nunique())
        if bins < 2:
            return pd.Series(0, index=s.index)
        return pd.qcut(s, q=bins, labels=False, duplicates="drop")

    df = df.copy()
    df["difficulty_bucket"] = qbin(df["difficulty"].astype(float))
    total = len(df)
    picks = []
    remaining = n_target
    groups = list(df.groupby(["domain", "difficulty_bucket"]))
    for i, (_, grp) in enumerate(groups):
        is_last = i == len(groups) - 1
        n = remaining if is_last else int(round(n_target * len(grp) / total))
        n = max(0, min(n, len(grp), remaining))
        if n > 0:
            chosen = rng.choice(grp.index.to_numpy(), size=n, replace=False)
            picks.append(df.loc[chosen])
            remaining -= n
    return pd.concat(picks).drop(columns=["difficulty_bucket"]) if picks else df.iloc[0:0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts-dir", type=str, default="scripts/rl_training_dataset/rollouts")
    p.add_argument("--candidates", type=str, default="scripts/rl_training_dataset/candidates.jsonl",
                   help="The prepare_candidates.py output, needed to recover full statement/reference text for survivors.")
    p.add_argument("--target-min", type=int, default=1000)
    p.add_argument("--target-max", type=int, default=1500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=str, default="scripts/rl_training_dataset/final_rl_training_set.jsonl")
    p.add_argument("--audit-output", type=str, default="scripts/rl_training_dataset/purged_theorems.jsonl",
                   help="Where to save the too-easy/too-hard theorems that got purged, for inspection.")
    return p.parse_args()


def main():
    args = parse_args()

    rows = load_rollout_rows(args.rollouts_dir)
    per_theorem = aggregate_by_theorem(rows)

    incomplete = per_theorem[per_theorem["n_variants"] < 2]
    if len(incomplete):
        print(f"[grade_and_filter] WARNING: {len(incomplete)} theorems only have rollouts for "
              f"one variant (pos or neg), not both -- their correct_count is out of "
              f"{incomplete['total_rollouts'].mode().tolist()} rollouts, not 8. Check that "
              f"generate_rollouts.py was run over the FULL candidates.jsonl (both variant rows "
              f"per theorem), not a filtered subset.")

    print_distribution(per_theorem, "correct_count", "CORRECT-COUNT DISTRIBUTION (per theorem, out of up to 8)")

    always_right = per_theorem["correct_count"] == per_theorem["total_rollouts"]
    always_wrong = per_theorem["correct_count"] == 0
    purged = per_theorem[always_right | always_wrong]
    survivors = per_theorem[~(always_right | always_wrong)].copy()

    print(f"\n[grade_and_filter] Theorems processed: {len(per_theorem)}")
    print(f"[grade_and_filter] Purged (always right, too easy): {int(always_right.sum())}")
    print(f"[grade_and_filter] Purged (always wrong, too hard):  {int(always_wrong.sum())}")
    print(f"[grade_and_filter] Survivors (mixed correctness):    {len(survivors)}")

    if len(survivors) > args.target_max:
        print(f"\n[grade_and_filter] {len(survivors)} survivors > target max {args.target_max}; "
              f"stratified-trimming down by domain x difficulty.")
        survivors = stratified_trim(survivors, args.target_max, args.seed)
        print(f"[grade_and_filter] Trimmed to {len(survivors)}.")
    elif len(survivors) < args.target_min:
        print(f"\n[grade_and_filter] WARNING: only {len(survivors)} survivors, below target min "
              f"{args.target_min}. Re-run prepare_candidates.py with a larger --n-candidates "
              f"and generate more rollouts, then re-run this script.")
    else:
        print(f"\n[grade_and_filter] {len(survivors)} survivors is within the "
              f"[{args.target_min}, {args.target_max}] target range. No trimming needed.")

    # recover full statement/label/reference text for surviving theorems from candidates.jsonl
    candidates = pd.read_json(args.candidates, lines=True)
    survivor_ids = set(survivors["base_id"])
    final_variants = candidates[candidates["base_id"].isin(survivor_ids)]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for _, row in final_variants.iterrows():
            f.write(json.dumps({
                "id": row["id"],
                "statement": row["statement"],
                "label": bool(row["label"]),
                "domain": row.get("domain"),
                "difficulty": row.get("difficulty"),
                "reference": row.get("reference"),
            }) + "\n")

    purged_ids = set(purged["base_id"])
    purged_variants = candidates[candidates["base_id"].isin(purged_ids)]
    os.makedirs(os.path.dirname(args.audit_output), exist_ok=True)
    purged_variants.merge(
        purged[["base_id", "correct_count", "total_rollouts"]], on="base_id"
    ).to_json(args.audit_output, orient="records", lines=True)

    print(f"\n[grade_and_filter] Final training set: {len(final_variants)} variant rows "
          f"({len(survivor_ids)} theorems) -> {args.output}")
    print(f"[grade_and_filter] Purged theorems (audit trail) -> {args.audit_output}")


if __name__ == "__main__":
    main()

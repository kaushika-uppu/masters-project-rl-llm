import argparse
import glob
import sys

import numpy as np
import pandas as pd


REQUIRED_COLS = {"Question_ID", "Score", "Word_Count", "Benchmark", "Model"}


def load_and_merge(input_patterns):
    files = []
    for pattern in input_patterns:
        files.extend(glob.glob(pattern))
    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No files matched any of: {input_patterns}")

    print(f"Loading {len(files)} CSV file(s):")
    for f in files:
        print(f"  - {f}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"{f} is missing required columns: {missing}")
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def normalize_word_count_per_model(df):
    # z-score word count within each model so different models are comparable
    def z(s):
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd > 0 else pd.Series(0.0, index=s.index)

    df = df.copy()
    df.loc[:, "Word_Count_Norm"] = df.groupby("Model")["Word_Count"].transform(z)
    return df


def aggregate_per_question(df):
    grouped = (
        df.groupby(["Benchmark", "Question_ID"])
        .agg(
            mean_score=("Score", "mean"),
            mean_word_count=("Word_Count", "mean"),
            mean_word_count_norm=("Word_Count_Norm", "mean"),
            n_models=("Model", "nunique"),
        )
        .reset_index()
    )
    return grouped


def assign_difficulty(df, score_weight=0.7):
    # difficulty combines incorrectness (1 - score) with normalized verbosity.
    # both are min-max scaled within benchmark so the weights are meaningful.
    df = df.copy()
    df.loc[:, "incorrectness"] = 1.0 - df["mean_score"]

    def minmax(s):
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else pd.Series(0.0, index=s.index)

    df.loc[:, "incorrectness_scaled"] = df.groupby("Benchmark")["incorrectness"].transform(minmax)
    df.loc[:, "verbosity_scaled"] = df.groupby("Benchmark")["mean_word_count_norm"].transform(minmax)
    df.loc[:, "difficulty"] = (
        score_weight * df["incorrectness_scaled"]
        + (1.0 - score_weight) * df["verbosity_scaled"]
    )
    return df


def bin_by_difficulty(df, n_bins):
    df = df.copy()

    def qbin(s):
        bins = min(n_bins, s.nunique())
        if bins < 2:
            return pd.Series(0, index=s.index)
        return pd.qcut(s, q=bins, labels=False, duplicates="drop")

    df.loc[:, "bucket"] = df.groupby("Benchmark")["difficulty"].transform(qbin)
    return df


def stratified_sample(df, fraction, seed):
    rng = np.random.default_rng(seed)
    picks = []
    for (_bench, _bucket), grp in df.groupby(["Benchmark", "bucket"]):
        n_target = int(round(len(grp) * fraction))
        n_target = max(1, min(n_target, len(grp)))
        chosen = rng.choice(grp.index.to_numpy(), size=n_target, replace=False)
        picks.append(df.loc[chosen])
    return pd.concat(picks).sort_values(["Benchmark", "Question_ID"]).reset_index(drop=True)


def print_distribution_check(full, sampled):
    print("\nDifficulty bucket distribution (full vs sampled):")
    for bench in sorted(full["Benchmark"].unique()):
        f = full[full["Benchmark"] == bench]
        s = sampled[sampled["Benchmark"] == bench]
        print(f"\n  Benchmark: {bench}  (full={len(f)}, sampled={len(s)})")
        print(f"  {'bucket':>6} {'full %':>10} {'sample %':>10} {'mean_score(full)':>20} {'mean_score(samp)':>20}")
        for b in sorted(f["bucket"].unique()):
            fb = f[f["bucket"] == b]
            sb = s[s["bucket"] == b]
            f_pct = 100 * len(fb) / len(f)
            s_pct = 100 * len(sb) / len(s) if len(s) else 0
            f_ms = fb["mean_score"].mean()
            s_ms = sb["mean_score"].mean() if len(sb) else float("nan")
            print(f"  {int(b):>6} {f_pct:>9.2f}% {s_pct:>9.2f}% {f_ms:>20.3f} {s_ms:>20.3f}")


def write_outputs(sampled, per_question, output_prefix):
    print("\nWrote:")
    for bench in sorted(sampled["Benchmark"].unique()):
        bench_sampled = sampled[sampled["Benchmark"] == bench].sort_values("Question_ID")
        bench_full = per_question[per_question["Benchmark"] == bench].sort_values("Question_ID")

        ids_path = f"{output_prefix}_{bench}_ids.txt"
        csv_path = f"{output_prefix}_{bench}_sampled.csv"
        full_path = f"{output_prefix}_{bench}_per_question.csv"

        with open(ids_path, "w") as f:
            for qid in bench_sampled["Question_ID"]:
                f.write(f"{qid}\n")

        bench_sampled.to_csv(csv_path, index=False)
        bench_full.to_csv(full_path, index=False)

        print(f"  [{bench}]")
        print(f"    {ids_path}            ({len(bench_sampled)} ids, one per line)")
        print(f"    {csv_path}       (sampled rows with difficulty info)")
        print(f"    {full_path}  (per-question aggregated stats for full benchmark)")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Merge multiple analytics CSVs (output of merge_results.py), normalize "
            "word counts per model, bin questions by difficulty, and produce a "
            "stratified-sampled list of representative IDs."
        )
    )
    p.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="One or more glob patterns matching analytics CSVs (e.g. 'outputs/*_analytics.csv').",
    )
    p.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="Prefix for output files (will produce <prefix>_ids.txt, <prefix>_sampled.csv, <prefix>_per_question.csv).",
    )
    p.add_argument(
        "--fraction",
        type=float,
        required=True,
        help="Fraction of the dataset to sample, e.g. 0.1 for 10%%.",
    )
    p.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of difficulty buckets per benchmark (default: 5).",
    )
    p.add_argument(
        "--score-weight",
        type=float,
        default=0.7,
        help="Weight on incorrectness vs verbosity in the difficulty score, in [0,1] (default: 0.7).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the stratified sampler (default: 42).",
    )
    args = p.parse_args()

    if not 0 < args.fraction <= 1:
        p.error("--fraction must be in (0, 1].")
    if not 0 <= args.score_weight <= 1:
        p.error("--score-weight must be in [0, 1].")
    if args.n_bins < 1:
        p.error("--n-bins must be >= 1.")

    return args


def main():
    args = parse_args()

    raw = load_and_merge(args.input)
    print(f"\nLoaded {len(raw)} rows across {raw['Model'].nunique()} model(s) "
          f"and {raw['Benchmark'].nunique()} benchmark(s).")

    raw = normalize_word_count_per_model(raw)
    per_question = aggregate_per_question(raw)
    per_question = assign_difficulty(per_question, score_weight=args.score_weight)
    per_question = bin_by_difficulty(per_question, n_bins=args.n_bins)

    sampled = stratified_sample(per_question, fraction=args.fraction, seed=args.seed)
    print(f"\nSampled {len(sampled)} of {len(per_question)} questions "
          f"({100 * len(sampled) / len(per_question):.2f}%).")

    print_distribution_check(per_question, sampled)
    write_outputs(sampled, per_question, args.output_prefix)


if __name__ == "__main__":
    main()

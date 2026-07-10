"""Summarize DeepTheorem judge eval runs."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = [
    "timestamp",
    "model_name",
    "checkpoint_path",
    "benchmark",
    "num_examples",
    "average_proof_score",
    "terminal_verdict_accuracy",
    "average_step_validity",
    "invalid_step_rate",
    "format_validity_rate",
    "results_file",
]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_result_file(
    results_file: str | Path,
    *,
    model_name: str,
    checkpoint_path: str = "",
    benchmark: str = "deeptheorem_judge",
) -> dict[str, Any]:
    path = Path(results_file)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    bench = data.get("results", {}).get(benchmark)
    if bench is None:
        names = list(data.get("results", {}).keys())
        raise ValueError(f"Benchmark '{benchmark}' not found in {path}. Available: {names}")

    rows = bench.get("results", [])
    verdicts: list[float] = []
    proofs: list[float] = []
    formats: list[float] = []
    bad_steps = 0
    steps = 0

    for row in rows:
        j = (row.get("prediction") or {}).get("judgement") or {}
        verdicts.append(float(j.get("verdict_correct", 0.0)))
        proofs.append(float(j.get("proof_validity", 0.0)))
        formats.append(float(j.get("format_validity", 0.0)))
        bad_steps += int(j.get("invalid_steps", 0))
        steps += int(j.get("valid_steps", 0)) + int(j.get("invalid_steps", 0))

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": model_name,
        "checkpoint_path": checkpoint_path,
        "benchmark": benchmark,
        "num_examples": int(bench.get("num_samples", len(rows))),
        "average_proof_score": float(bench.get("metrics", {}).get("average_score", 0.0)),
        "terminal_verdict_accuracy": _mean(verdicts),
        "average_step_validity": _mean(proofs),
        "invalid_step_rate": bad_steps / steps if steps else 0.0,
        "format_validity_rate": _mean(formats),
        "results_file": str(path),
    }


def append_summary(summary_csv: str | Path, row: dict[str, Any]) -> None:
    path = Path(summary_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def parse_args():
    parser = argparse.ArgumentParser(description="Track DeepTheorem verifier evaluation progress.")
    parser.add_argument("results_file", help="Evaluation JSON file to summarize.")
    parser.add_argument("--summary-csv", default="evaluation/results/deeptheorem/summary.csv")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint-path", default="")
    parser.add_argument("--benchmark", default="deeptheorem_judge")
    return parser.parse_args()


def main():
    args = parse_args()
    row = summarize_result_file(
        args.results_file,
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        benchmark=args.benchmark,
    )
    append_summary(args.summary_csv, row)
    print(f"Wrote summary row to {args.summary_csv}")


if __name__ == "__main__":
    main()

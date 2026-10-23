#!/usr/bin/env python3
"""Measure how reliable the RL step judge is.

Examples:
  python scripts/eval_judge.py --stub --problems data/dt_eval_dataset.jsonl --max-problems 20
  python scripts/eval_judge.py --judge-model Qwen/Qwen2.5-7B-Instruct,Qwen/Qwen2.5-32B-Instruct
  python scripts/eval_judge.py --labeled-file my_labels.jsonl --mode labeled --no-4bit

torch/transformers are imported lazily, only when a real judge is loaded.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.training.rl  # noqa: E402,F401  (import first: avoids a pre-existing src.judge <-> src.training.rl import cycle)
from src.judge import judge_eval as je  # noqa: E402
from src.judge.step_judge import DeterministicJudge  # noqa: E402
from src.training.rl.problems import load_problems_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def training_problem_files() -> set:
    names = set()
    for cfg in glob.glob(str(ROOT / "src/training/configs/*.yaml")):
        try:
            for m in re.finditer(r"problems_path:\s*[\"']?([^\"'\s#]+)", Path(cfg).read_text()):
                names.add(os.path.basename(m.group(1)))
        except OSError:
            pass
    return names


def looks_like_training_file(path: str) -> bool:
    base = os.path.basename(path)
    return base in training_problem_files() or bool(re.search(r"rl_problems|rl_subset|rl_train", base))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default="data/dt_eval_dataset.jsonl")
    ap.add_argument("--max-problems", type=int, default=200)
    ap.add_argument("--n-per-problem", type=int, default=2, help="gold positives per problem")
    ap.add_argument("--corruptions", default="all", help="comma list of: " + ",".join(je.CORRUPTION_KINDS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--judge-model", action="append", default=None,
                    help="HF model id; repeat or comma-separate to compare judges")
    ap.add_argument("--load-in-4bit", dest="load_in_4bit", action="store_true", default=True)
    ap.add_argument("--no-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--labeled-file", default=None)
    ap.add_argument("--mode", choices=["synthetic", "labeled", "both"], default=None)
    ap.add_argument("--stub", action="store_true", help="DeterministicJudge, no GPU (smoke test)")
    ap.add_argument("--output-dir", default=None)
    return ap.parse_args(argv)


def build_items(args):
    mode = args.mode or ("labeled" if args.labeled_file else "synthetic")
    if mode in ("labeled", "both") and not args.labeled_file:
        raise SystemExit("--mode %s requires --labeled-file" % mode)
    items, stats = [], {}
    if mode in ("synthetic", "both"):
        if looks_like_training_file(args.problems):
            print(f"WARNING: {args.problems} looks like an RL TRAINING problems file; "
                  "use a held-out set for an unbiased judge evaluation.", file=sys.stderr)
        if not os.path.exists(args.problems):
            raise SystemExit(f"problems file not found: {args.problems}")
        problems = load_problems_jsonl(args.problems)[: args.max_problems]
        kinds = None if args.corruptions.strip().lower() == "all" else \
            [k.strip() for k in args.corruptions.split(",") if k.strip()]
        syn, stats = je.build_eval_items_with_stats(problems, args.n_per_problem, kinds, args.seed)
        items += syn
        print(f"[eval_judge] problems={stats['n_problems']} usable={stats['n_usable']} "
              f"skipped_no_reference={stats['n_skipped_no_reference']} "
              f"single_step_refs={stats['n_single_step_references']} items={stats['n_items']}")
        if any(stats["skipped_by_kind"].values()):
            print(f"[eval_judge] skipped per kind: {stats['skipped_by_kind']}")
    if mode in ("labeled", "both"):
        lab = je.load_labeled_items(args.labeled_file)
        print(f"[eval_judge] labeled items: {len(lab)}")
        items += lab
    if not items:
        raise SystemExit("no evaluation items could be built")
    return items, stats


def load_real_judge(name, args):
    from src.training.rl.grpo_tree import _load_judge_model  # lazy: pulls transformers
    from src.training.rl.inproc import TransformersJudge

    model, tok = _load_judge_model({
        "judge_model": name,
        "judge_load_in_4bit": args.load_in_4bit,
        "judge_device": args.device,
    })
    model.eval()
    return TransformersJudge(model, tok, max_new_tokens=args.max_new_tokens), model


def free(judge, model):
    del judge, model
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def evaluate_one(name, judge, items, args, out_dir: Path, stats):
    out_dir.mkdir(parents=True, exist_ok=True)
    judgements, raws = je.run_judge(judge, items, args.batch_size)
    metrics = je.compute_metrics(items, judgements)
    metrics["judge"] = name
    metrics["eval_config"] = {k: getattr(args, k) for k in
                              ("problems", "max_problems", "n_per_problem", "corruptions", "seed",
                               "load_in_4bit", "max_new_tokens", "mode", "labeled_file")}
    metrics["build_stats"] = stats
    with open(out_dir / "items.jsonl", "w") as f:
        for it, j, raw in zip(items, judgements, raws):
            row = it.to_dict()
            row.update(raw_output=raw,
                       judgement={"valid": j.valid, "reason": j.reason, "is_terminal": j.is_terminal,
                                  "verdict": j.verdict, "state_summary": j.state_summary},
                       correct=bool(j.valid) == it.gold_valid)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report = je.format_report(metrics, title=f"Judge reliability: {name}")
    (out_dir / "report.txt").write_text(report + "\n")
    print(report + "\n")
    return metrics


def main(argv=None) -> int:
    args = parse_args(argv)
    items, stats = build_items(args)

    names = []
    for v in args.judge_model or ["Qwen/Qwen2.5-7B-Instruct"]:
        names += [x.strip() for x in v.split(",") if x.strip()]
    if args.stub:
        names = ["stub"]

    quant = "4bit" if args.load_in_4bit else "bf16"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = lambda n: "stub" if n == "stub" else f"{slugify(n)}_{quant}"  # noqa: E731
    base = Path(args.output_dir) if args.output_dir else None
    multi = len(names) > 1
    if base is None:
        base = ROOT / "results/judge_eval" / (f"compare_{ts}" if multi else f"{tag(names[0])}_{ts}")

    rows = {}
    for name in names:
        out_dir = base / tag(name) if multi else base
        if name == "stub":
            rows[name] = evaluate_one(name, DeterministicJudge(), items, args, out_dir, stats)
        else:
            print(f"[eval_judge] loading {name} ({quant}) on {args.device}")
            judge, model = load_real_judge(name, args)
            try:
                rows[tag(name)] = evaluate_one(name, judge, items, args, out_dir, stats)
            finally:
                free(judge, model)
    base.mkdir(parents=True, exist_ok=True)
    comp = je.format_comparison(rows)
    (base / "comparison.txt").write_text(comp + "\n")
    print(comp)
    print(f"[eval_judge] wrote results to {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

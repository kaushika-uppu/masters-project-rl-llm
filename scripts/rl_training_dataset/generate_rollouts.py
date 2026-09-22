"""Phase 2 of the RL training-set build: run rollouts on the candidate variant
rows produced by prepare_candidates.py, using vLLM directly (same pattern as
scripts/deeptheorem_phase1.py / deeptheorem_phase2.py), so this can run as a
SLURM array job on the cluster.

Each input row is ONE variant (pos or neg) of a theorem. This script generates
--n-rollouts (default 4) completions per row. Since prepare_candidates.py
emits both the pos and neg row for every sampled theorem, running this over
both gives you 4 pos-variant + 4 neg-variant rollouts per theorem = 8 total,
exactly the "8 rollouts of each question (4 pos, 4 neg)" the final dataset
needs for grade_and_filter.py to score correct_count out of 8.

PROMPT / VERDICT FORMAT
------------------------
Uses the <step>...</step> + \\boxed{proved|disproved} convention from the
prompt-inconsistency fix on miranda/sft_v2 (evaluation/benchmarks/
deeptheorem_eval.py), not the older "Verdict: PROVED/DISPROVED" line used in
src/data/deeptheorem.py's PROVE_OR_DISPROVE_SYSTEM_PROMPT. grade_and_filter.py
accepts both anyway (ported from the same fix), but the rollout PROMPT here
asks for \\boxed{...} specifically since that's the convention Miranda's
current branch is moving to.

Chunking (--chunk-id/--chunk-size) matches the existing phase1/phase2 scripts
so this can be run as a SLURM array job -- see the sample sbatch template in
scripts/rl_training_dataset/README.md.
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

# System prompt shared with the fixed eval convention.
DEEPTHEOREM_PROVE_OR_DISPROVE_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. Provide step-by-step proofs "
    "for mathematical theorems. Break down your reasoning into clear, logical steps."
)


def build_user_prompt(statement: str) -> str:
    return (
        f"{statement}\n\n"
        "Provide a rigorous step-by-step proof or counterexample. "
        "Wrap each reasoning step in <step>...</step>. "
        r"End with the verdict in the same format as the examples: "
        r"\boxed{proved} or \boxed{disproved}."
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=str, default="scripts/rl_training_dataset/candidates.jsonl")
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--chunk-id", type=int, required=True)
    p.add_argument("--chunk-size", type=int, default=500)
    p.add_argument("--n-rollouts", type=int, default=4,
                   help="Rollouts PER VARIANT ROW. 4 here + running both pos and neg rows "
                        "for a theorem gives 8 total rollouts per theorem.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--output-dir", type=str, default="scripts/rl_training_dataset/rollouts")
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Cannot find input '{args.input}'. Run prepare_candidates.py first.")

    df = pd.read_json(args.input, lines=True)

    start_idx = args.chunk_id * args.chunk_size
    end_idx = start_idx + args.chunk_size
    chunk_df = df.iloc[start_idx:end_idx].copy()

    if chunk_df.empty:
        print(f"Chunk {args.chunk_id} is out of bounds (input has {len(df)} rows). Exiting.")
        return

    print(f"Processing rows {start_idx} to {end_idx} of {len(df)} with model={args.model} ...")

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        n=args.n_rollouts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    tokenizer = llm.get_tokenizer()
    prompts = []
    for statement in chunk_df["statement"].tolist():
        messages = [
            {"role": "system", "content": DEEPTHEOREM_PROVE_OR_DISPROVE_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(str(statement))},
        ]
        prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    print(f"Generating {args.n_rollouts} rollouts per row for {len(prompts)} rows...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        row = chunk_df.iloc[i]
        rollouts = [out.text for out in output.outputs]
        results.append({
            "id": row["id"],
            "base_id": row["base_id"],
            "variant": row["variant"],
            "statement": row["statement"],
            "label": bool(row["label"]),
            "domain": row.get("domain"),
            "difficulty": row.get("difficulty"),
            "rollouts": rollouts,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"rollouts_chunk_{args.chunk_id}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} rows x {args.n_rollouts} rollouts each to {output_file}")


if __name__ == "__main__":
    main()

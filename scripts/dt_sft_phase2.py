import argparse
import json
import os
import pandas as pd
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"ERROR: Cannot find the dataset file at '{args.input}'.")

    with open(args.input, 'r') as f:
        df = pd.read_json(f, lines=True)
    
    start_idx = args.chunk_id * args.chunk_size
    end_idx = start_idx + args.chunk_size
    chunk_df = df.iloc[start_idx:end_idx].copy()
    
    if chunk_df.empty:
        print(f"Chunk {args.chunk_id} is out of bounds. Exiting.")
        return

    print(f"Processing SFT Phase 2 chunk {args.chunk_id} (rows {start_idx} to {end_idx})...")

    llm = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        tensor_parallel_size=2,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        enforce_eager=True
    )

    # 8 rollouts per question
    sampling_params = SamplingParams(
        n=8,
        temperature=0.6,
        max_tokens=6000
    )

    tokenizer = llm.get_tokenizer()
    prompts = []

    for _, row in chunk_df.iterrows():
        question = str(row.get('tested_variant', ''))
        
        system_instruction = "You are an expert mathematical reasoning assistant. Please logically prove or disprove the following theorem. You must show your step-by-step reasoning and conclude with either \\boxed{True} or \\boxed{False}."
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": question}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(formatted_prompt)

    print(f"Generating 8 rollouts per question for {len(prompts)} theorems...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        row_data = chunk_df.iloc[i].to_dict()
        rollouts = [out.text for out in output.outputs]
        
        row_data["sft_rollouts"] = rollouts
        results.append(row_data)

    os.makedirs("results/sft_phase2", exist_ok=True)
    output_file = f"results/sft_phase2/sft_p2_rollouts_chunk_{args.chunk_id}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved {len(results)} evaluated SFT variants to {output_file}")

if __name__ == "__main__":
    main()
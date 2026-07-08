import argparse
import pandas as pd
from vllm import LLM, SamplingParams
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="deeptheorem_stratified_30k.jsonl")
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()

    df = pd.read_json(args.input, lines=True)
    
    start_idx = args.chunk_id * args.chunk_size
    end_idx = start_idx + args.chunk_size
    chunk_df = df.iloc[start_idx:end_idx].copy()
    
    if chunk_df.empty:
        print(f"Chunk {args.chunk_id} is out of bounds. Exiting.")
        return

    print(f"Processing questions {start_idx} to {end_idx}...")

    llm = LLM(
        model="Qwen/Qwen2.5-Math-7B-Instruct",
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.90
    )

    # need to generate 4 distinct rollouts at once
    sampling_params = SamplingParams(
        n=4,
        temperature=0.7,
        max_tokens=2048
    )

    tokenizer = llm.get_tokenizer()
    prompts = []

    for q in chunk_df['ori_question'].tolist():
        messages = [
            {"role": "system", "content": "You are an expert mathematical reasoning assistant. Please logically prove or disprove the following theorem."},
            {"role": "user", "content": str(q)}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        prompts.append(formatted_prompt)

    print(f"Generating 4 rollouts per question for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        row = chunk_df.iloc[i]
        
        question = row.get('ori_question', '')
        response = row.get('ori_solution', '')
        truth_value = row.get('truth_value', None)
        root_topic = row.get('root_topic', None)
        difficulty = row.get('difficulty', None)
        
        # extracting 4 generated answers
        rollouts = [out.text for out in output.outputs]
        
        results.append({
            "id": int(row.name),
            "ori_question": question,
            "ori_solution": response,            
            "truth_value": truth_value,          
            "root_topic": root_topic,                    
            "difficulty": difficulty,
            "rollouts": rollouts
        })

    os.makedirs("phase1_results", exist_ok=True)
    output_file = f"phase1_results/rollouts_chunk_{args.chunk_id}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved {len(results)} questions with 4 rollouts each to {output_file}")

if __name__ == "__main__":
    main()
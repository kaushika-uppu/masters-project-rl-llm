import argparse
import pandas as pd
from vllm import LLM, SamplingParams
import json
import os
import ast

def extract_variant(val, default_truth):
    """Parses the nested JSON variant and extracts both question text and explicit truth_value."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
        
    # parse string-encoded JSON or lists
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('[') or val.startswith('{'):
            try:
                val = ast.literal_eval(val)
            except:
                try:
                    val = json.loads(val)
                except:
                    pass

    if isinstance(val, list) and len(val) > 0:
        val = val[0]

    if isinstance(val, dict):
        question_text = str(val.get('question', val.get('ori_question', ''))).strip()
        
        if not question_text:
            return None
            
        # get explicit truth value, fallback to default if somehow missing
        truth = val.get('truth_value', default_truth)
        if isinstance(truth, str):
            truth = truth.strip().lower() == 'true'
            
        return {"question": question_text, "truth": bool(truth)}

    # fallback
    if isinstance(val, str) and val.strip() != "":
        return {"question": val.strip(), "truth": default_truth}
        
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=1000)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"ERROR: Cannot find the dataset file at '{args.input}'. Please check your file paths.")
        
    with open(args.input, 'r') as f:
        df = pd.read_json(f, lines=True)
    
    start_idx = args.chunk_id * args.chunk_size
    end_idx = start_idx + args.chunk_size
    chunk_df = df.iloc[start_idx:end_idx].copy()
    
    if chunk_df.empty:
        print(f"Chunk {args.chunk_id} is out of bounds. Exiting.")
        return

    print(f"Processing chunk {args.chunk_id} (rows {start_idx} to {end_idx}).")

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
    evaluation_targets = []

    for row_idx, row in chunk_df.iterrows():
        # True as default for pos, False as default for neg
        pos_data = extract_variant(row.get('pos'), default_truth=True)
        neg_data = extract_variant(row.get('neg'), default_truth=False)
        
        tests_to_run = []
        if pos_data:
            tests_to_run.append({"question": pos_data["question"], "truth": pos_data["truth"], "type": "positive"})
        if neg_data:
            tests_to_run.append({"question": neg_data["question"], "truth": neg_data["truth"], "type": "negative"})
            
        # fallback to original question if both variants are broken/missing
        if not tests_to_run:
            tests_to_run.append({
                "question": str(row.get('ori_question', '')), 
                "truth": str(row.get('truth_value'), None), 
                "type": "original"
            })

        for test in tests_to_run:
            evaluation_targets.append({
                "id": int(row.name),
                "ori_question": str(row.get('ori_question', '')),
                "ori_solution": str(row.get('ori_solution', '')),
                "root_topic": row.get('root_topic', None),                    
                "difficulty": float(row.get('difficulty')) if pd.notna(row.get('difficulty')) else None,
                "tested_variant": test["question"],
                "truth_value": test["truth"],
                "variant_type": test["type"]
            })

            messages = [
                {
                    "role": "system", 
                    "content": "You are an expert mathematical reasoning assistant. Please logically prove or disprove the following theorem. You must conclude your reasoning with either \\boxed{True} or \\boxed{False}."
                },
                {"role": "user", "content": test["question"]}
            ]
            
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(formatted_prompt)

    print(f"Generating 4 rollouts per question for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        target_data = evaluation_targets[i]
        rollouts = [out.text for out in output.outputs]
        target_data["rollouts"] = rollouts
        results.append(target_data)

    os.makedirs("phase1_results", exist_ok=True)
    output_file = f"phase1_results/rollouts_chunk_{args.chunk_id}.json"
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved {len(results)} evaluated variants to {output_file}")

if __name__ == "__main__":
    main()
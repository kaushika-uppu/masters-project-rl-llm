import json
import os
import pandas as pd
from datasets import load_dataset
import ast

def main():
    sft_file = "src/training/dt_sft_stratified_30k.jsonl"
    rl_file = "src/training/dt_stratified_30k.jsonl"
    output_file = "data/dt_complete_eval_dataset.jsonl"
    
    used_ids = set()
    for file_path in [sft_file, rl_file]:
        if os.path.exists(file_path):
            print(f"Reading IDs from {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        used_ids.add(json.loads(line)['id'])
        else:
            print(f"WARNING: {file_path} not found. Ensure the path is correct.")

    print(f"Total unique used IDs collected: {len(used_ids)}")

    print("Loading full DeepTheorem dataset from Hugging Face...")
    dataset = load_dataset("Jiahao004/DeepTheorem", split="train")
    df = dataset.to_pandas()
    
    # filter out the used questions
    unused_df = df[~df['id'].isin(used_ids)].copy()
    print(f"Unused questions available: {len(unused_df)}")
    
    if len(unused_df) == 0:
        print("ERROR: No unused questions left!")
        return

    # sample 333 unique theorem groups
    sample_size = min(333, len(unused_df))
    eval_sample = unused_df.sample(n=sample_size, random_state=42)
    
    print(f"Sampled {sample_size} theorems. Expanding into Original, Pos, and Neg variants...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    eval_questions_count = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for _, row in eval_sample.iterrows():
            base_id = row['id']
            domain = str(row.get('domain', ''))
            difficulty = row.get('difficulty', 0.0)
            
            # dataset stores original theorem here
            original_q = row.get('informal_theorem_qa')
            
            def parse_variant(variant_data):
                if isinstance(variant_data, str):
                    try:
                        return ast.literal_eval(variant_data)
                    except:
                        return None
                return variant_data
            
            pos_data = parse_variant(row.get('pos'))
            neg_data = parse_variant(row.get('neg'))
            
            # original theorem
            if original_q:
                ori_out = {
                    "id": f"{base_id}_ori",
                    "base_id": base_id,                 # grouping key
                    "variant_type": "original",         # identifies criteria role
                    "statement": original_q,
                    "label": True,                      # original theorems are true
                    "domain": domain,
                    "difficulty": difficulty
                }
                outfile.write(json.dumps(ori_out) + "\n")
                eval_questions_count += 1
            
            # entailing variant (positive)
            if pos_data and 'question' in pos_data:
                pos_out = {
                    "id": f"{base_id}_pos",
                    "base_id": base_id,                 # grouping key
                    "variant_type": "entailing",        # identifies criteria role
                    "statement": pos_data["question"],
                    "label": True,                      # entailing must be true
                    "domain": domain,
                    "difficulty": difficulty
                }
                outfile.write(json.dumps(pos_out) + "\n")
                eval_questions_count += 1
                
            # contradictory variant (negative)
            if neg_data and 'question' in neg_data:
                neg_out = {
                    "id": f"{base_id}_neg",
                    "base_id": base_id,                 # grouping key
                    "variant_type": "contradictory",    # identifies criteria role
                    "statement": neg_data["question"],
                    "label": False,                     # contradictory must be false
                    "domain": domain,
                    "difficulty": difficulty
                }
                outfile.write(json.dumps(neg_out) + "\n")
                eval_questions_count += 1

    print("\n" + "="*50)
    print("COMPLETE EVALUATION DATASET CREATED SUCCESSFULLY!")
    print(f"Total Eval Questions: {eval_questions_count} (Grouped by base_id)")
    print(f"Saved to: {output_file}")
    print("="*50)

if __name__ == "__main__":
    main()
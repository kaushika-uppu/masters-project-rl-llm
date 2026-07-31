import json
import os
import ast
import re
from datasets import load_dataset

def normalize_text(text):
    if not text: return ""
    return re.sub(r'\s+', '', text).lower()

def main():
    subset_file = "src/training/jsonl_data/dt_sft_subset_1_2.jsonl"
    output_file = "data/sft_dt_eval_dataset.jsonl"
    
    dataset = load_dataset("Jiahao004/DeepTheorem", split="train")
    
    hf_lookup = {}
    print("Building text-to-original-ID mapping...")
    
    for row in dataset:
        ori_q = row.get('ori_question', '')
        if ori_q:
            norm_text = normalize_text(ori_q)
            hf_lookup[norm_text] = {
                'original_id': row['id'],
                'original_domain': row.get('domain', '')
            }
            
    print(f"Mapped {len(hf_lookup)} original theorems.")

    # build the eval dataset
    print(f"\nProcessing subset file: {subset_file}...")
    
    if not os.path.exists(subset_file):
        print(f"ERROR: {subset_file} not found!")
        return

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    eval_questions_count = 0
    missing_count = 0
    
    with open(subset_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
             
        for line in infile:
            if not line.strip(): continue
            row = json.loads(line)
            
            subset_ori_q = row.get('ori_question', '')
            norm_q = normalize_text(subset_ori_q)
            
            if norm_q in hf_lookup:
                orig_id = hf_lookup[norm_q]['original_id']
                orig_domain = hf_lookup[norm_q]['original_domain']
            else:
                missing_count += 1
                continue
                
            pos_data = row.get('pos', {})
            neg_data = row.get('neg', {})
            difficulty = row.get('difficulty', 0.0)
                
            # positive variant
            if 'question' in pos_data:
                pos_out = {
                    "id": f"{orig_id}_pos",
                    "statement": pos_data["question"],
                    "label": True,
                    "domain": orig_domain,
                    "difficulty": difficulty
                }
                outfile.write(json.dumps(pos_out) + "\n")
                eval_questions_count += 1
                
            # negative variant
            if 'question' in neg_data:
                neg_out = {
                    "id": f"{orig_id}_neg",
                    "statement": neg_data["question"],
                    "label": False,
                    "domain": orig_domain,
                    "difficulty": difficulty
                }
                outfile.write(json.dumps(neg_out) + "\n")
                eval_questions_count += 1

    print("\n" + "="*50)
    print("EVALUATION DATASET CREATED SUCCESSFULLY!")
    print(f"Total Eval Questions Generated: {eval_questions_count} (Pos + Neg)")
    if missing_count > 0:
        print(f"Note: {missing_count} questions could not be matched to HF and were skipped.")
    print(f"Saved to: {output_file}")
    print("="*50)

if __name__ == "__main__":
    main()
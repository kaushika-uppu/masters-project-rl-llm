import json
import os

def process_file_pair(stratified_file, input_file, output_file):
    print(f"\n--- Processing {input_file} ---")
    print(f"Reading data from {stratified_file}...")
    
    # build mapping of ori_question -> {proof, informal_theorem, informal_theorem_qa}
    data_map = {}
    try:
        with open(stratified_file, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                ori_q = row.get("ori_question")
                proof = row.get("proof")
                informal_theorem = row.get("informal_theorem")
                informal_theorem_qa = row.get("informal_theorem_qa")
                
                if ori_q and proof is not None and informal_theorem is not None and informal_theorem_qa is not None:
                    data_map[ori_q] = {
                        "proof": proof,
                        "informal_theorem": informal_theorem,
                        "informal_theorem_qa": informal_theorem_qa
                    }
    except FileNotFoundError:
        print(f"ERROR: Could not find {stratified_file}. Skipping...")
        return

    print(f"Loaded {len(data_map)} unique mappings into memory.")

    # inject data into input file
    print(f"Injecting proofs and informal theorems into {input_file}...")
    matched = 0
    missed = 0
    duplicates_skipped = 0
    seen_questions = set()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                row = json.loads(line)
                ori_q = row.get("ori_question")

                # make sure it's not a duplicate question
                if ori_q in seen_questions:
                    duplicates_skipped += 1
                    continue

                seen_questions.add(ori_q)
                
                # map both proof and informal_theorem
                if ori_q in data_map:
                    row["proof"] = data_map[ori_q]["proof"]
                    row["informal_theorem"] = data_map[ori_q]["informal_theorem"]
                    row["informal_theorem_qa"] = data_map[ori_q]["informal_theorem_qa"]
                    matched += 1
                else:
                    row["proof"] = ""
                    row["informal_theorem"] = ""
                    row["informal_theorem_qa"] = ""
                    missed += 1
                    
                # drop rollouts columns from data curation step
                if "sft_rollouts" in row:
                    del row["sft_rollouts"]
                if "rollouts" in row:
                    del row["rollouts"]

                outfile.write(json.dumps(row) + "\n")
                
    except FileNotFoundError:
        print(f"ERROR: Could not find {input_file}. Skipping...")
        return

    print(f"Successfully matched: {matched}")
    if missed > 0:
        print(f"WARNING: Missed {missed} rows (no matching data found)")
    if duplicates_skipped > 0:
        print(f"DEDUPLICATION: Skipped {duplicates_skipped} duplicate rows based on 'ori_question'")
    print(f"Saved new dataset to: {output_file}")


def main():
    tasks = [
        {
            "stratified": "src/training/dt_sft_stratified_30k.jsonl",
            "input": "src/training/dt_sft_phase2_full.jsonl",
            "output": "src/training/dt_sft_data.jsonl"
        },
        {
            "stratified": "src/training/dt_stratified_30k.jsonl",
            "input": "src/training/dt_phase2_full.jsonl",
            "output": "src/training/dt_rl_data.jsonl"
        }
    ]
    
    for task in tasks:
        process_file_pair(task["stratified"], task["input"], task["output"])
        
    print("\n" + "="*50)
    print("ALL DATASETS GENERATED SUCCESSFULLY!")
    print("="*50)

if __name__ == "__main__":
    main()
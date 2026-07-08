import json
import glob
import re
import pandas as pd
import os
import argparse

def extract_boxed_truth(text):
    """
    Parses for \boxed{True} or \boxed{False} ignoring case.
    """
    match = re.search(r'\\boxed\{.{0,30}?(True|False).{0,30}?\}', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).capitalize() == 'True'
    return None

def main():
    parser = argparse.ArgumentParser(description="Grade Phase 1 Rollouts")
    parser.add_argument("--sft", action="store_true", help="Run grader for SFT pipeline")
    args = parser.parse_args()

    if args.sft:
        print("Loading SFT Phase 1 Rollouts...")
        input_pattern = "results/sft_phase1_results/*.json"
        output_name = "./src/training/dt_sft_phase1_survivors.jsonl"
    else:
        print("Loading RL Phase 1 Rollouts...")
        input_pattern = "results/dt_phase1/*.json"
        output_name = "./src/training/dt_phase1_survivors.jsonl"

    file_list = glob.glob(input_pattern)
    
    if not file_list:
        print("No JSON files found in 'results/dt_phase1/'.")
        return

    survivors = []
    total_processed = 0
    total_purged = 0

    for file_path in file_list:
        with open(file_path, 'r') as f:
            chunk_data = json.load(f)
            
        for item in chunk_data:
            total_processed += 1
            ground_truth = item.get('truth_value')
            
            if ground_truth is None:
                continue
                
            correct_count = 0
            for rollout in item.get('rollouts', []):
                model_prediction = extract_boxed_truth(rollout)
                
                # compare extracted prediction against native truth value
                if model_prediction == ground_truth:
                    correct_count += 1

            # keep question if model didn't get it right at all
            if correct_count > 0:
                total_purged += 1 # purge >0 correct (too easy)
            else:
                survivors.append(item)
                    

    print("\n" + "=" * 50)
    print("PHASE 1 PURGE COMPLETE")
    print(f"Total Variants Evaluated: {total_processed}")
    print(f"Too Easy (Purged):        {total_purged}")
    print(f"Hard Variants (Kept):     {len(survivors)}")
    print("=" * 50 + "\n")

    # save surviving dataset
    if survivors:
        df_survivors = pd.DataFrame(survivors)
        df_survivors = df_survivors.drop(columns=['rollouts'], errors='ignore')

        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        df_survivors.to_json(output_name, orient="records", lines=True)

        print(f"Saved surviving dataset to: {output_name}")
        
    else:
        print("No questions survived the purge.")

if __name__ == "__main__":
    main()
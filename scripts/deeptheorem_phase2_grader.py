import os
import json
import glob
import re
from collections import Counter
import argparse

def extract_boxed_truth(text):
    """Parses for \boxed{True} or \boxed{False} ignoring case."""
    match = re.search(r'\\boxed\{.{0,30}?(True|False).{0,30}?\}', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).capitalize()
    return None

def main():
    parser = argparse.ArgumentParser(description="Grade Phase 2 Rollouts")
    parser.add_argument("--sft", action="store_true", help="Run grader for SFT pipeline")
    args = parser.parse_args()

    if args.sft:
        print("=== SFT MODE ===")
        input_dir = "results/sft_phase2"
        output_file = "src/training/dt_sft_phase2_survivors.jsonl"
        rollout_key = "sft_rollouts"
        min_keep = 1
        max_keep = 2
    else:
        print("=== RL MODE ===")
        input_dir = "results/dt_phase2"
        output_file = "src/training/dt_phase2_survivors.jsonl"
        rollout_key = "rollouts"
        min_keep = 2
        max_keep = 5

    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    file_pattern = os.path.join(input_dir, "*.json")
    files = glob.glob(file_pattern)
    
    if not files:
        print(f"ERROR: No chunked JSON files found in {input_dir}/")
        return
        
    print(f"Found {len(files)} chunk files.")
    print(f"1. Injecting scores into original chunk files.")
    print(f"2. Filtering Questions with {min_keep}-{max_keep} correct into {output_file}.\n")
    
    total_questions = 0
    max_rollouts = 0
    score_counts = Counter()
    goldilocks_count = 0
    
    with open(output_file, 'w') as out_f:
        for file_path in sorted(files):
            filename = os.path.basename(file_path)
            
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse {filename}. Skipping.")
                    continue
            
            for row in data:
                total_questions += 1
                
                # extract ground truth
                expected = str(row.get('truth_value', None)).capitalize()
                    
                rollouts = row.get(rollout_key, [])
                if len(rollouts) > max_rollouts:
                    max_rollouts = len(rollouts)

                correct_count = 0
                for rollout in rollouts:
                    pred = extract_boxed_truth(rollout)
                    if pred == expected:
                        correct_count += 1
                
                # inject score directly into row dictionary
                row['correct_count'] = correct_count
                score_counts[correct_count] += 1
                
                # filter and save to JSONL
                if min_keep <= correct_count <= max_keep:
                    out_f.write(json.dumps(row) + '\n')
                    goldilocks_count += 1
            
            # overwrite original chunk file with updated data
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f" Scored {filename} & extracted survivors")
            
    print("\n" + "=" * 65)
    print("=== FINAL PHASE 2 DATASET DISTRIBUTION ===")
    print(f"Pipeline: {'SFT' if args.sft else 'RL'}")
    print(f"Total Questions Processed: {total_questions}")
    print("-" * 65)
    print("Score (Correct) | Question Count | Visual Distribution")
    print("-" * 65)
    
    for i in range(max_rollouts + 1):
        count = score_counts.get(i, 0)
        percentage = (count / total_questions) * 100 if total_questions > 0 else 0
        bar = "█" * int(percentage / 2) 
        print(f"{i}/{max_rollouts} correct   | {count:14d} | {bar} ({percentage:.1f}%)")

    print("-" * 65)
    print(f"Goldilocks Zone ({min_keep}-{max_keep}): {goldilocks_count} questions.")
    print(f"Too Hard (<{min_keep}):          {sum(score_counts[i] for i in range(0, min_keep))} questions dropped.")
    print(f"Too Easy (>{max_keep}):         {sum(score_counts[i] for i in range(max_keep + 1, max_rollouts + 1))} questions dropped.")
    print("=" * 65)
    print(f"Grading complete. Saved questions to: ./{output_file}")

if __name__ == "__main__":
    main()
import glob
import json
import argparse
import csv
import os

def grade_strict_evaluation(input_pattern, benchmark_name, output_csv, output_json):
    file_list = glob.glob(input_pattern)
    if not file_list:
        print(f"Error: No files found matching '{input_pattern}'")
        return
    
    print(f"Loading {len(file_list)} result files...")

    # group all predictions by their base theorem ID
    theorems = {}
    
    # flat list to store all combined questions for the merged JSON
    all_merged_questions = []
    master_metadata = None
    
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if master_metadata is None:
                master_metadata = data.get("metadata", {})
                
            questions = data.get("results", {}).get(benchmark_name, {}).get("results", [])
            
            for item in questions:
                all_merged_questions.append(item)
                
                meta = item.get("metadata", {})
                base_id = str(meta.get("base_id", ""))
                variant = meta.get("variant_type", "")
                
                if not base_id or not variant:
                    continue
                
                if base_id not in theorems:
                    theorems[base_id] = {
                        "original": [],
                        "entailing": [],
                        "contradictory": []
                    }
                    
                theorems[base_id][variant].append({
                    "id": item.get("id"),
                    "prediction": item.get("prediction"),
                    "text": item.get("generated_text", "")
                })

    strict_correct = 0
    total_theorems = 0
    all_groups_analysis = []
    
    for base_id, variants in theorems.items():
        # ensure there's at least one of each variant to grade the group
        if not variants["original"] or not variants["entailing"] or not variants["contradictory"]:
            continue
            
        total_theorems += 1
        
        # extract predictions
        ori_preds = [v["prediction"] for v in variants["original"]]
        ent_preds = [v["prediction"] for v in variants["entailing"]]
        con_preds = [v["prediction"] for v in variants["contradictory"]]
        
        # combine all predictions for Criteria 1 check
        all_preds = ori_preds + ent_preds + con_preds
        
        # --- evaluate the 4 criteria ---
        crit_1_pass = None not in all_preds
        crit_2_pass = all(p is True for p in ori_preds)
        crit_3_pass = all(p is True for p in ent_preds)
        crit_4_pass = all(p is False for p in con_preds)
        
        overall_pass = crit_1_pass and crit_2_pass and crit_3_pass and crit_4_pass
        
        if overall_pass:
            strict_correct += 1

        all_groups_analysis.append({
            "base_id": base_id,
            "overall_pass": overall_pass,
            "crit_1_valid_formats": crit_1_pass,
            "crit_2_original_pass": crit_2_pass,
            "crit_3_entailing_pass": crit_3_pass,
            "crit_4_contradictory_pass": crit_4_pass,
            "ori_preds": ori_preds,
            "ent_preds": ent_preds,
            "con_preds": con_preds
        })

    accuracy = strict_correct / total_theorems if total_theorems > 0 else 0

    # export to CSV
    if all_groups_analysis:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_groups_analysis[0].keys())
            writer.writeheader()
            writer.writerows(all_groups_analysis)

    # export to JSON
    if all_merged_questions:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
        final_json_output = {
            "metadata": master_metadata,
            "results": {
                benchmark_name: {
                    "num_samples": len(all_merged_questions),
                    "results": all_merged_questions
                }
            }
        }
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(final_json_output, f, indent=2)

    print("\n" + "=" * 60)
    print("4-CRITERIA EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Complete Theorem Groups   : {total_theorems}")
    print(f"Theorems Passing All 4 Criteria : {strict_correct}")
    print("-" * 60)
    print(f"ACCURACY                 : {accuracy * 100:.2f}%")
    print("=" * 60)
    print(f"Saved merged JSON (all {len(all_merged_questions)} questions) to: {output_json}")
    print(f"Saved CSV summary (all {total_theorems} groups) to    : {output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Pattern to find evaluation JSON files (e.g. 'evaluation/results/eval_sft/deeptheorem/*.json')")
    parser.add_argument("--benchmark", default="deeptheorem", help="Benchmark key")
    parser.add_argument("--output-csv", required =True, help="Path to save summary CSV")
    parser.add_argument("--output-json", required=True, help="Path to save merged JSON")
    args = parser.parse_args()
    
    grade_strict_evaluation(args.input, args.benchmark, args.output_csv, args.output_json)
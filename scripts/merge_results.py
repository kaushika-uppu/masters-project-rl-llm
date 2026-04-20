import argparse
import json
import glob
import os

# function to combine all the json files from different jobs (created when benchmark evaluation run on HPC)
def merge_benchmark_results(input_pattern, output_filename, benchmark_name):
    file_list = glob.glob(input_pattern)

    if not file_list:
        print(f"Error: No files found matching '{input_pattern}'")
        return
    
    print(f"Found {len(file_list)} JSON files matching '{input_pattern}'.")

    global_total_samples = 0
    global_total_correct = 0
    merged_results_list = []

    master_metadata = None

    for file_path in file_list:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
            if master_metadata is None:
                master_metadata = data.get("metadata", {})
            
            if benchmark_name not in data.get("results", {}):
                print(f"Warning: Benchmark '{benchmark_name}' not found in {file_path}. Skipping.")
                continue
                
            bench_data = data["results"][benchmark_name]
            
            global_total_samples += bench_data.get("num_samples", 0)
            metrics = bench_data.get("metrics", {})
            global_total_correct += metrics.get("total_correct", 0)
            
            if "results" in bench_data:
                merged_results_list.extend(bench_data["results"])

    final_accuracy = global_total_correct / global_total_samples if global_total_samples > 0 else 0

    # sorting results in order of ID
    merged_results_list.sort(key=lambda x: x.get("id", 0))

    final_output = {
        "metadata": master_metadata,
        "results": {
            benchmark_name: {
                "num_samples": global_total_samples,
                "metrics": {
                    "total_correct": global_total_correct,
                    "accuracy": final_accuracy,
                    "average_score": final_accuracy
                },
                "results": merged_results_list
            }
        }
    }
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2)

    print("=" * 70)
    print("EVALUATION SUMMARY:")
    print(f"Benchmark:                 {benchmark_name}")
    print(f"Total Questions:           {global_total_samples}")
    print(f"Total Correct:             {global_total_correct}")
    print(f"FINAL ACCURACY:            {final_accuracy * 100:.2f}%")
    print("-" * 70)
    print(f"Saved to: {output_filename}")

def parse_args():
    parser = argparse.ArgumentParser(description="Merge distributed evaluation JSON files.")

    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="File name pattern to find the JSON files (e.g., 'results_*.json' or 'outputs/gsm8k/*.json')"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Name of the final merged JSON file to save results to"
    )
    
    parser.add_argument(
        "--benchmark", 
        type=str, 
        required=True, 
        help="Name of the benchmark inside the JSON structure"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    merge_benchmark_results(args.input, args.output, args.benchmark)
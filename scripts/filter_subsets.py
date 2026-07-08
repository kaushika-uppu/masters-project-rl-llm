import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Filter Master Dataset by Correct Rollouts")
    parser.add_argument("--input", type=str, required=True, help="Path to master .jsonl file")
    parser.add_argument("--output", type=str, required=True, help="Path to save filtered .jsonl file")
    parser.add_argument("--min", type=int, required=True, help="Minimum correct rollouts")
    parser.add_argument("--max", type=int, required=True, help="Maximum correct rollouts")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Cannot find {args.input}")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    kept_count = 0
    
    print(f"Filtering {args.input} for scores between {args.min} and {args.max}...")
    
    with open(args.input, 'r') as infile, open(args.output, 'w') as outfile:
        for line in infile:
            row = json.loads(line)
            score = row.get('correct_count', 0)
            
            if args.min <= score <= args.max:
                outfile.write(line)
                kept_count += 1
                
    print(f"Extracted {kept_count} questions.")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
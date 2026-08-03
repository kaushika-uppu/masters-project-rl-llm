import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def main():
    input_file = "data/jsonl_data/dt_rl_subset_5_7.jsonl"
    output_file = "data/jsonl_data/dt_rl_subset_5_7_1K.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading dataset from {input_file}...")
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    df = pd.DataFrame(records)
    print(f"Total records loaded: {len(df)}")

    try:
        print("Performing stratified sampling...")
        train_df, remaining_df = train_test_split(
            df, 
            train_size=1000, 
            stratify=df['root_topic'], 
            random_state=42
        )
    except ValueError as e:
        print("Warning: A topic has too few examples to stratify. Falling back to random sampling.")
        train_df = df.sample(n=1000, random_state=42)

    print(f"Saving {len(train_df)} questions to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in train_df.to_dict(orient='records'):
            f.write(json.dumps(record) + "\n")

    print("\n" + "="*50)
    print(f"Training dataset saved: {output_file}")
    
    print("\nOriginal Topic Distribution vs New 1,000 Subset:")
    orig_dist = df['root_topic'].value_counts(normalize=True) * 100
    new_dist = train_df['root_topic'].value_counts(normalize=True) * 100
    
    comparison = pd.DataFrame({'Original %': orig_dist, 'New %': new_dist})
    print(comparison.round(1).to_string())
    print("="*50)

if __name__ == "__main__":
    main()
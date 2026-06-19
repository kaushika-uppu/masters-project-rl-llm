import pandas as pd
import numpy as np
from datasets import load_dataset
import ast

# topics are in lists in "domain" column, need to extract overall topic from it
def extract_root_topic(topic_data):
    """
    Extracts the highest-level mathematical domain from the nested topic annotations.
    e.g., 'Algebra -> Abstract Algebra -> Field Theory' becomes 'Algebra'
    """
    # mapping typos/super detailed domain names
    TOPIC_MAPPING = {
    "Linear algebra": "Linear Algebra",
    "linear Algebra  linear transformations / conv definition": "Linear Algebra",
    "Discrete Math": "Discrete Mathematics",
    "Advanced Calculus": "Calculus",
    "Higher Algebra Sub-Topics;Intermediate Algebra;Functional Analysis;-": "Abstract Algebra",
    "Higher Algebra Sub-Topics;Intermediate Algebra;Functional Analysis;": "Abstract Algebra",
    "Statistics": "Applied Mathematics",
    "Set Theory": "Discrete Mathematics"
    }
    
    if isinstance(topic_data, np.ndarray):
        topic_data = topic_data.tolist()
            
    if isinstance(topic_data, str) and topic_data.startswith('['):
        try:
            topic_data = ast.literal_eval(topic_data)
        except (ValueError, SyntaxError):
            topic_data = [topic_data.strip("[]")]
            
    if isinstance(topic_data, list) and len(topic_data) > 0:
        first_topic = str(topic_data[0])
        raw_topic = first_topic.split('->')[0].strip(' "\'')
    elif isinstance(topic_data, str):
        raw_topic = topic_data.split('->')[0].strip(' "\'')

    if raw_topic in TOPIC_MAPPING:
        return TOPIC_MAPPING[raw_topic]
    if "analysis" in raw_topic.lower():
        return "Analysis"
        
    return raw_topic if raw_topic != "Unknown" else "Other"

def main():
    print("Loading dataset")
    dataset = load_dataset("Jiahao004/DeepTheorem", split="train")
    df = dataset.to_pandas()

    df['root_topic'] = df['domain'].apply(extract_root_topic)
    print("Sample of extracted Root Topics:")
    print(df[['domain', 'root_topic']].head(3))
    
    # EDA
    print("\n--- Exploratory Data Analysis ---")
    
    # difficulty distribution
    print("\nQuestion counts by Difficulty Level:")
    difficulty_counts = df['difficulty'].value_counts().sort_index()
    print(difficulty_counts)
    
    # topic distribution
    print("\nQuestion counts by Topic Category:")
    topic_counts = df['root_topic'].value_counts() 
    print(topic_counts)
    
    # how topics are distributed across difficulties
    print("\nCross-Tabulation: Topic vs. Difficulty:")
    cross_tab = pd.crosstab(df['root_topic'], df['difficulty'])
    print(cross_tab)


    # stratified sampling dataset reduction
    target_size = 30000
    print(f"\n--- Reducing dataset to {target_size} samples ---")
    # only using questions with level 7.0 difficulty or above
    df = df[df['difficulty'] >= 7.0].copy()
    
    # makes sure topics with low numbers have at least 1 question
    def safe_stratified_sample(x):
        n_samples = int(round(len(x) / len(df) * target_size))
        # guarantee at least 1 sample so micro-topics don't get erased
        n_samples = max(1, n_samples)
        n_samples = min(n_samples, len(x))
        
        return x.sample(n=n_samples, random_state=42)

    reduced_df = df.groupby(['root_topic', 'difficulty'], group_keys=False).apply(safe_stratified_sample)
    
    remaining_needed = target_size - len(reduced_df)
    
    if remaining_needed > 0:
        leftovers = df[~df.index.isin(reduced_df.index)].sample(n=remaining_needed, random_state=42)
        reduced_df = pd.concat([reduced_df, leftovers])
    elif remaining_needed < 0:
        reduced_df = reduced_df.sample(n=target_size, random_state=42)
        
    reduced_df = reduced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print("\nFinal Reduced Dataset Distribution (Root Topic vs Difficulty):")
    print(pd.crosstab(reduced_df['root_topic'], reduced_df['difficulty']))
    
    output_file = "dt_stratified_30k.jsonl"
    reduced_df.to_json(output_file, orient="records", lines=True)
    print(f"\nSuccessfully saved reduced dataset to {output_file}")

if __name__ == "__main__":
    main()
from datasets import load_dataset, Dataset
from typing import Literal

DatasetName = Literal["deeptheorem", "gsm8k"]

def get_dataset(dataset: DatasetName) -> Dataset:
    if dataset == "deeptheorem":
        return load_dataset("Jiahao004/DeepTheorem", split="train")
    
    if dataset == "gsm8k":
        return load_dataset("openai/gsm8k", "main", split="train")
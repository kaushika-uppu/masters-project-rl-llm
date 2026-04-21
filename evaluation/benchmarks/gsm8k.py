from evaluation.benchmarks.base_benchmark import BaseBenchmark, DataSetItem
from typing import List
from datasets import load_dataset
import re

class GSM8K(BaseBenchmark[str, str]):
    def load_dataset(self) -> List[DataSetItem[str, str]]:
        dataset = load_dataset("openai/gsm8k", "main", num_proc=1, split="test")
        items = []
        for i, item in enumerate(dataset):
            # gsm8k benchmark has reasoning + answer lumped together; need to extract answer
            ground_truth = item["answer"].split("#### ")[-1].strip()
            items.append(
                DataSetItem(
                    id = i,
                    input = item["question"],
                    output = ground_truth,
                    metadata = {"original answer": item["answer"]}
                )
            )
        return items
    
    def get_user_prompt(self, input: str) -> str:
        return f"""You are an expert mathematician. Please solve the following grade-school math problem.
        
        Read the problem carefully, break down the logic step-by-step, and perform the calculations accurately.
        
        Problem: {input}
        
        Please provide your response strictly in the following format:
        <reasoning>
        [Write your step-by-step reasoning and mathematical calculations here]
        </reasoning>
        Place your final numeric answer strictly within \\boxed{{}}.
        For example: \\boxed{{42}}
        """
    
    def parse_output(self, output: str) -> str:
        answer_match = re.search(r'\\boxed\{([^}]+)\}', output)

        if answer_match:
            return answer_match.group(1).strip()
        return ""
    
    def score(self, item: DataSetItem[str, str], at_output: str) -> float:
        if item.output is None:
            return 0.0
        
        # need to clean math output before evaluating
        def clean_number(text: str) -> str:
            text = text.replace(",", "")
            match = re.search(r'-?\d+(?:\.\d+)?', text)
            if match:
                num_str = match.group()
                if num_str.endswith('.0'):
                    num_str = num_str[:-2]
                return num_str
            return text
        
        cleaned_actual = clean_number(at_output)
        cleaned_pred = clean_number(item.output)
        return 1.0 if cleaned_pred == cleaned_actual else 0.0
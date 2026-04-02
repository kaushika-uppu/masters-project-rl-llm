from evaluation.benchmarks.base_benchmark import BaseBenchmark, DataSetItem
from typing import List
from datasets import load_dataset
import re

class MATH500(BaseBenchmark[str, str]):
    def load_dataset(self) -> List(DataSetItem[str, str]):
        """
        Item format:
        {
            "problem": string,
            "solution": string,
            "answer": string,
            "subject": string,
            "level": int,
            "unique_id": string
        }
        """
        dataset = load_dataset("HuggingFaceH4/MATH-500", num_proc=1)
        return [DataSetItem(
            input=item["problem"], 
            output=item["answer"], 
            id=item["unique_id"],
            metadata={
                "subject": item["subject"],
                "level": item["level"]
            })
            for item in dataset["train"]]

    def get_user_prompt(self, input: str) -> str:
        # prompt comes from https://www.vals.ai/benchmarks/math500 which defines a way to prompt the LLM rather than fine-tune for LaTeX
        # if a model doesn't have prior LaTeX knowledge, it may perform worse than it should but I do not believe this will be an issue
        return f"""
        Answer the following math question, given in LaTeX format, clearly and concisely, and present the final answer as \\(\\boxed{{x}}\\), where x is the fully simplified solution.

        Example:
        **Question:** \\(\\int_0^1 (3x^2 + 2x) \\,dx\\)
        **Solution:** \\(\\int (3x^2 + 2x) \\,dx = x^3 + x^2 + C\\) Evaluating from 0 to 1: \\((1^3 + 1^2) - (0^3 + 0^2) = 1 + 1 - 0 = 2 \\boxed{{2}}\\)

        Now, solve the following question: {input}
        """
    
    def parse_output(self, output: str) -> str:
        answer_match = re.search(r'\\boxed{\s*(.*?)\s*}', output, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return answer_match(1), strip()
        return ""
    
    def score(self, item: DataSetItem[str, str], at_output: str) -> float:
        if item.output is None:
            return 0.0
        return 1.0 if item.output.lower() == at_output.lower() else 0.0


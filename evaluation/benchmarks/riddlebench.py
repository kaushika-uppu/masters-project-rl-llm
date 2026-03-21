from evaluation.benchmarks.base_benchmark import BaseBenchmark, DataSetItem
from typing import List
from datasets import load_dataset
import re

class RiddleBench(BaseBenchmark[str, str]):
    def load_dataset(self) -> List[DataSetItem[str, str]]:
        """
        Example Entry:
        {
            "id": 1051,
            "type": "coding and decoding sum",
            "question": "If 'CARING' is coded as 'EDVGKC', and 'SHARES' is coded as 'UKEPBO', then how will 'CASKET' be coded as in the same code? a) EDXIBP c) EDWPAI b) EDWIAP d) EDWIBP",
            "answer": "d"
        }
        """
        dataset = load_dataset("ai4bharat/RiddleBench", num_proc=1)
        return [DataSetItem(input=item["question"], output=item["answer"]) for item in dataset["train"]]

    def get_user_prompt(self, input: str) -> str:
        return f""" {input}

        Please provide your response in the following format:
        <reasoning>
        [Your step-by-step reasoning here]
        </reasoning>
        <answer>
        [Your final answer here as specified in the question]
        </answer>
        """

    def parse_output(self, output: str) -> str:
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        return ""
        
    def score(self, gt_output: str, at_output: str) -> float:
        return 1.0 if gt_output.lower() == at_output.lower() else 0.0
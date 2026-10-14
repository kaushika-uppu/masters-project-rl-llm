from evaluation.benchmarks.base_benchmark import BaseBenchmark, DataSetItem
from typing import List
from datasets import load_dataset
import re

# Tolerant matcher for a boxed answer. Accepts all of:
#   \boxed{42}   boxed{42}   Boxed{42}   $\boxed{42}$   <boxed{42}>   .Boxed{ 42 }
# i.e. the leading backslash is OPTIONAL and the keyword is case-insensitive. RL
# checkpoints frequently drop the backslash, which the old `r'\\boxed\{...\}'` regex
# silently scored 0 (prediction == ""), under-reporting accuracy.
_BOXED_RE = re.compile(r"\\?boxed\s*\{\s*([^}]*?)\s*\}", re.IGNORECASE | re.DOTALL)

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
        return [DataSetItem(input=item["question"], output=item["answer"], id=item["id"], metadata={"type": item["type"]}) for item in dataset["train"]]

    def get_user_prompt(self, input: str) -> str:
        return f""" {input}

        Please provide your response in the following format:
        <reasoning>
        [Your step-by-step reasoning here]
        </reasoning>
        Place your final numeric answer strictly within \\boxed{{}}.
        For example: \\boxed{{42}}
        """

    def parse_output(self, output: str) -> str:
        """Extract the final boxed answer, tolerating a missing backslash / case / $ / tags.

        Takes the LAST match (models sometimes restate earlier candidates) and strips
        surrounding LaTeX/whitespace so `\\boxed{ 42 }` and `$\\boxed{42}$` both yield
        `42`. Returns "" when there is no boxed answer at all (scored 0, as before).
        """
        if not output:
            return ""
        matches = _BOXED_RE.findall(output)
        if not matches:
            return ""
        return matches[-1].strip().strip("$").strip()
        
    def score(self, item: DataSetItem[str, str], at_output: str) -> float:
        if item.output is None:
            return 0.0
        return 1.0 if item.output.lower() == at_output.lower() else 0.0
    
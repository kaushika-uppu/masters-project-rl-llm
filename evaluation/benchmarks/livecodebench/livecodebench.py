from evaluation.benchmarks.base_benchmark import BaseBenchmark, DataSetItem
from typing import List
from datasets import load_dataset
from dataclasses import dataclass
import re
from .utils import parse_public_tests, parse_private_tests, format_examples, evaluate_problem

# note: the most rest cases you use the slower this gets
TEST_CASES_TO_USE = 10

@dataclass
class CodeProblem:
    question: str
    tests: dict  # Should be dict with "input" and "output" keys

class LiveCodeBench(BaseBenchmark[CodeProblem, str]):
    def load_dataset(self) -> List[DataSetItem[CodeProblem, str]] :
        dataset = load_dataset("livecodebench/code_generation_lite", version_tag="release_v2", num_proc=1, split="test", trust_remote_code=True)

        return [
            DataSetItem(
                input=CodeProblem(
                    question=item["question_content"],
                    tests=parse_public_tests(item["public_test_cases"])
                ),
                id=item["question_id"],
                output=None,
                metadata={
                    "private_tests": parse_private_tests(item["private_test_cases"], TEST_CASES_TO_USE),
                    "difficulty": item["difficulty"]
                }
            )
            for item in dataset
        ][:10]

    def get_user_prompt(self, input: CodeProblem) -> str:
        return f"""
            You are an expert programmer. Please solve the following problem.
            Read the problem carefully and write a Python solution that reads from stdin and writes to stdout.

            Problem: {input.question}

            Test Cases:
            {format_examples(input.tests)}

            Write your solution between <python> and </python> tags:
            <python>
            # your solution here
            </python>
        """

    def parse_output(self, output: str) -> str:
        answer_match = re.search(r'<python>\s*(.*?)\s*</python>', output, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        return ""

    def score(self, item: DataSetItem, predicted: str) -> float:
        if not predicted or predicted.strip() == "":
            return 0.0

        private_tests = item.metadata.get("private_tests")
        if not private_tests:
            private_tests = item.input.tests

        if not private_tests or not private_tests.get("input"):
            return 0.0

        results = evaluate_problem(predicted, private_tests)

        if not results:
            return 0.0

        return 1.0 if all(results) else 0.0


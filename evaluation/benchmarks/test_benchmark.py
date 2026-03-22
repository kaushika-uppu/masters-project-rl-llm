from evaluation.benchmarks.base_benchmark import BaseBenchmark, DataSetItem
from typing import List

class TestBenchmark(BaseBenchmark[str, str]):
    def load_dataset(self) -> List[DataSetItem[str, str]]:
        return [
            DataSetItem(input="test_correct", output="test_correct"),
            DataSetItem(input="test_incorrect", output="incorrect")
        ]

    def get_user_prompt(self, input: str) -> str:
        return input
    
    def parse_output(self, output: str) -> str:
        return output
    
    def score(self, item: DataSetItem[str, str], at_output: str) -> float:
        return 1.0 if item.output == at_output else 0.0
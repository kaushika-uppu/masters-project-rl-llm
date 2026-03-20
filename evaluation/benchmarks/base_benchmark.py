# evaluation/benchmarks/base_benchmark.py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, List
from dataclasses import dataclass


# Define type variables for input and output types
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')

@dataclass
class DataSetItem(Generic[InputType, OutputType]):
    """A single item in a benchmark dataset."""
    input: InputType
    output: OutputType

class BaseBenchmark(ABC, Generic[InputType, OutputType]):
    """
    Base class for all benchmarks.
    
    Type parameters:
        InputType: Type of the input data for the benchmark.
        OutputType: Type of the output data for the benchmark.
    """

    @abstractmethod
    def load_dataset(self) -> List[DataSetItem[InputType, OutputType]]:
        pass

    @abstractmethod
    def get_user_prompt(self, input: InputType) -> str:
        pass

    @abstractmethod
    def parse_output(self, output: str) -> OutputType:
        pass

    @abstractmethod
    def score(self, gt_output: OutputType, at_output: OutputType) -> float:
        pass

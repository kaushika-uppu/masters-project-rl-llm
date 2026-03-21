"""Benchmark implementations."""

from evaluation.benchmarks.base_benchmark import BaseBenchmark, DataSetItem
from evaluation.benchmarks.test_benchmark import TestBenchmark
from evaluation.benchmarks.riddlebench import RiddleBench

# Export base class
__all__ = ["BaseBenchmark", "DataSetItem", "TestBenchmark", "RiddleBench"]

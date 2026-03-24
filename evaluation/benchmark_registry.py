# evaluation/benchmark_registry.py
# keeps track of all available benchmarks

from typing import Dict, List, Type
from evaluation.benchmarks import BaseBenchmark, TestBenchmark, RiddleBench, LiveCodeBench

BENCHMARK_REGISTRY: Dict[str, Type[BaseBenchmark]]= {
    "test": TestBenchmark,
    "riddlebench": RiddleBench,
    "livecodebench": LiveCodeBench,
}

def get_benchmarks() -> List[str]:
    """Return a list of available benchmark names."""
    return list(BENCHMARK_REGISTRY.keys())

def create_benchmark(name: str) -> BaseBenchmark:
    """Create a benchmark instance by name."""
    if name not in BENCHMARK_REGISTRY:
        available = get_benchmarks()
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    
    return BENCHMARK_REGISTRY[name]()
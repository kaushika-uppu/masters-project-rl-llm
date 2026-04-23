# evaluation/benchmark_registry.py
# keeps track of all available benchmarks

from typing import Dict, List, Type
from evaluation.benchmarks import BaseBenchmark, TestBenchmark, RiddleBench, LiveCodeBench, GSM8K, MATH500

BENCHMARK_REGISTRY: Dict[str, Type[BaseBenchmark]]= {
    "test": TestBenchmark,
    "riddlebench": RiddleBench,
    "livecodebench": LiveCodeBench,
    "gsm8k": GSM8K,
    "math500": MATH500
}

# Dataset sizes for each benchmark (used for SLURM job array sizing)
BENCHMARK_DATASET_SIZES: Dict[str, int] = {
    "test": 2,
    "riddlebench": 1737,
    "livecodebench": 511,
    "gsm8k": 1319,
    "math500": 500
}

def get_benchmarks() -> List[str]:
    """Return a list of available benchmark names."""
    return list(BENCHMARK_REGISTRY.keys())

def get_benchmark_size(name: str) -> int:
    """Return the dataset size for a benchmark."""
    if name not in BENCHMARK_DATASET_SIZES:
        available = get_benchmarks()
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    return BENCHMARK_DATASET_SIZES[name]

def create_benchmark(name: str) -> BaseBenchmark:
    """Create a benchmark instance by name."""
    if name not in BENCHMARK_REGISTRY:
        available = get_benchmarks()
        raise ValueError(f"Unknown benchmark '{name}'. Available: {available}")
    return BENCHMARK_REGISTRY[name]()
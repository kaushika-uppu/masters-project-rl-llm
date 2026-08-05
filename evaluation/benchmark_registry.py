# evaluation/benchmark_registry.py
# keeps track of all available benchmarks

from importlib import import_module
from typing import Dict, List
from evaluation.benchmarks.base_benchmark import BaseBenchmark

BENCHMARK_REGISTRY: Dict[str, str] = {
    "test": "evaluation.benchmarks.test_benchmark:TestBenchmark",
    "riddlebench": "evaluation.benchmarks.riddlebench:RiddleBench",
    "livecodebench": "evaluation.benchmarks.livecodebench.livecodebench:LiveCodeBench",
    "gsm8k": "evaluation.benchmarks.gsm8k:GSM8K",
    "math500": "evaluation.benchmarks.math_500.math_500:MATH500",
    "deeptheorem": "evaluation.benchmarks.deeptheorem_eval:DeepTheoremEval",
    "deeptheorem_judge": "evaluation.benchmarks.deeptheorem_judge:DeepTheoremJudgeEval",
}

# Dataset sizes for each benchmark (used for SLURM job array sizing)
BENCHMARK_DATASET_SIZES: Dict[str, int] = {
    "test": 2,
    "riddlebench": 1737,
    "livecodebench": 511,
    "gsm8k": 1319,
    "math500": 500,
    "deeptheorem": 999,
    "deeptheorem_judge": 1000,
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
    module_name, class_name = BENCHMARK_REGISTRY[name].split(":")
    module = import_module(module_name)
    return getattr(module, class_name)()

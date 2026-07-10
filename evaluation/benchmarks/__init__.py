from evaluation.benchmarks.base_benchmark import BaseBenchmark, DataSetItem

__all__ = [
    "BaseBenchmark",
    "DataSetItem",
    "TestBenchmark",
    "RiddleBench",
    "LiveCodeBench",
    "GSM8K",
    "MATH500",
    "DeepTheoremEval",
    "DeepTheoremJudgeEval",
]


def __getattr__(name):
    if name == "TestBenchmark":
        from evaluation.benchmarks.test_benchmark import TestBenchmark
        return TestBenchmark
    if name == "RiddleBench":
        from evaluation.benchmarks.riddlebench import RiddleBench
        return RiddleBench
    if name == "LiveCodeBench":
        from evaluation.benchmarks.livecodebench.livecodebench import LiveCodeBench
        return LiveCodeBench
    if name == "GSM8K":
        from evaluation.benchmarks.gsm8k import GSM8K
        return GSM8K
    if name == "MATH500":
        from evaluation.benchmarks.math_500.math_500 import MATH500
        return MATH500
    if name == "DeepTheoremEval":
        from evaluation.benchmarks.deeptheorem_eval import DeepTheoremEval
        return DeepTheoremEval
    if name == "DeepTheoremJudgeEval":
        from evaluation.benchmarks.deeptheorem_judge import DeepTheoremJudgeEval
        return DeepTheoremJudgeEval
    raise AttributeError(name)

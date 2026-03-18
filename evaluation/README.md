# Evaluation Framework

## Adding a New Benchmark

1. Create benchmark file** in `benchmarks/`
2. Register in `benchmark_registry.py`
3. Export in `benchmarks/__init__.py`

Reference the `test_benchmark.py` file and where it is registered / exported for examples.

## Usage

```bash
python scripts/evaluate.py --inference-fn dummy --benchmarks my_benchmark --verbose
```


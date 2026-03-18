# Source Code

## Adding a New Inference Function

1. Create inference file** in `inference/`
2. Register in `function_registry.py`
3. Export in `inference/__init__.py`

Reference the `test_inference.py` file and where it is registered / exported for examples.

## Usage

```bash
python scripts/evaluate.py --inference-fn my_model --benchmarks test --verbose
```

## Notes

- All inference functions must have signature: `(prompt: str) -> str`


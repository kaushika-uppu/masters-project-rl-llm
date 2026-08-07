# Post-Training LLMs Using Reinforcement Learning To Develop Critical Thinking

Training and evaluating LLMs using Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT) with chain-of-thought (CoT) prompting capabilities.

## Project Structure

```
├── src/                   # Core source code
│   ├── models/            # Model wrappers and loading
│   ├── inference/         # Inference and CoT prompting
│   ├── judge/             # Step judge prompts and verifier helpers
│   ├── training/          # SFT and RL training
│   └── utils/             # Shared utilities
├── evaluation/            # Evaluation framework
│   ├── benchmarks/        # Benchmark implementations
│   ├── deeptheorem/       # DeepTheorem eval utilities
│   ├── results/           # Evaluation results (gitignored)
│   └── trimmed_ids/       # Benchmark subset id lists
├── configs/               # Configuration files
├── scripts/               # Executable scripts
├── tests/                 # Unit tests
├── data/                  # Datasets (gitignored)
└── checkpoints/           # Model checkpoints (gitignored)
```

## Setup

1. Virtual environment (optional)
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If you get a script execution policy error, run:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating again.

2. Install requirements
```bash
pip install -r requirements.txt
```

## Running Evaluation

```bash
python scripts/evaluate.py --inference-fn <function_name> --benchmarks <benchmark_names> [--workers N] [--verbose]
```

**Example:**
```bash
python scripts/evaluate.py --inference-fn dummy --benchmarks test --verbose
```

**Options:**
- `--inference-fn`: Inference function to use (see `src/function_registry.py`)
- `--benchmarks`: Space-separated list of benchmarks (see `evaluation/benchmark_registry.py`), if not provided, all available benchmarks are used
- `--workers`: Number of parallel workers (default: 1)
- `--output-dir`: Results directory (default: `evaluation/results`)
- `--verbose`: Show progress bars and detailed output
- `--limit`: Running only a select number of examples

"""RL entry dispatcher, wired from src/training/run.py.

config['rl']['method']:
  "grpo_baseline" (default) - trajectory-level GRPO, endpoint reward (proves infra)
  "grpo_tree"               - tree + per-step MC advantages + novelty (the contribution)
"""

from __future__ import annotations

from .baseline_grpo import run_grpo_baseline
from .grpo_tree import run_grpo_tree


def run_rl(config: dict, model, tokenizer) -> None:
    rl_cfg = config["rl"]
    method = rl_cfg.get("method", "grpo_baseline")
    print(f"[rl] method = {method}")
    if method == "grpo_baseline":
        run_grpo_baseline(model, tokenizer, rl_cfg)
    elif method == "grpo_tree":
        run_grpo_tree(model, tokenizer, rl_cfg)
    else:
        raise ValueError(
            f"Unknown rl.method '{method}'. Use 'grpo_baseline' or 'grpo_tree'."
        )

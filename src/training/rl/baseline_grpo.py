"""Baseline GRPO (trajectory-level, endpoint reward) via TRL.

The de-risking baseline: no tree, no process reward — just the verifiable verdict reward
over full completions. Runs on a PROVIDED list of examples (jsonl); question selection is
external. Supports TRL's native vLLM generation (`use_vllm`), the standard high-throughput
path on clusters, and `num_generations` controls the group size.
"""

from __future__ import annotations

import re

from src.data.deeptheorem import PROVE_OR_DISPROVE_SYSTEM_PROMPT

from .problems import load_problems_jsonl

# trl and datasets are imported lazily inside run_grpo_baseline so this module stays
# importable without those heavy/optional deps installed.

_VERDICT = re.compile(r"verdict\s*:?\s*(proved|disproved)", re.IGNORECASE)


def _verdict(text: str):
    m = _VERDICT.findall(text or "")
    return m[-1].lower() if m else None


def run_grpo_baseline(model, tokenizer, rl_cfg: dict):
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    problems = load_problems_jsonl(
        rl_cfg["problems_path"], limit=rl_cfg.get("max_problems")
    )
    rows = [
        {
            "prompt": [
                {"role": "system", "content": PROVE_OR_DISPROVE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Prove or disprove the following:\n{p.statement}",
                },
            ],
            "label": p.label,
        }
        for p in problems
    ]
    train_ds = Dataset.from_list(rows)

    def reward_verdict(completions, label, **kwargs):
        scores = []
        for comp, lab in zip(completions, label):
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            truth = "proved" if lab else "disproved"
            scores.append(1.0 if _verdict(text) == truth else 0.0)
        return scores

    args_kw = dict(
        output_dir=rl_cfg["output_dir"],
        num_generations=rl_cfg.get("num_generations", 8),
        per_device_train_batch_size=rl_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=rl_cfg.get("gradient_accumulation_steps", 1),
        learning_rate=rl_cfg.get("learning_rate", 1e-6),
        num_train_epochs=rl_cfg.get("num_train_epochs", 1),
        max_completion_length=rl_cfg.get("max_completion_length", 512),
        logging_steps=rl_cfg.get("logging_steps", 10),
        save_steps=rl_cfg.get("save_steps", 200),
        bf16=rl_cfg.get("bf16", True),
        report_to=rl_cfg.get("report_to", "none"),
    )
    # vLLM generation (standard cluster throughput path) — only pass if requested,
    # since older TRL versions may not accept these kwargs.
    if rl_cfg.get("use_vllm"):
        args_kw["use_vllm"] = True
        if "vllm_gpu_memory_utilization" in rl_cfg:
            args_kw["vllm_gpu_memory_utilization"] = rl_cfg[
                "vllm_gpu_memory_utilization"
            ]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_verdict,
        args=GRPOConfig(**args_kw),
        train_dataset=train_ds,
    )
    trainer.train()
    trainer.save_model(rl_cfg["output_dir"])
    print(f"[grpo_baseline] done -> {rl_cfg['output_dir']}")

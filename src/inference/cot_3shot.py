"""
Runs 3-shot Chain-of-Thought prompting with Qwen2.5-7B-Instruct.
Loads fixed examples from a JSON file, builds the final prompt, runs inference,
and returns the model's generated answer text.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
import yaml

from src.models.qwen_wrapper import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Prompt file for the current benchmark (riddlebench)
PROMPT_FILE = PROJECT_ROOT / "prompts" / "cot_3shot" / "riddlebench.json"
CONFIG_FILE = PROJECT_ROOT / "configs" / "qwen2.5_7b.yaml"

_prompt_data: Optional[dict[str, Any]] = None
_config_data: Optional[dict[str, Any]] = None
_model = None
_tokenizer = None

def _load_prompt_data() -> dict[str, Any]:
    """Load prompt config once."""
    global _prompt_data

    if _prompt_data is not None:
        return _prompt_data

    with PROMPT_FILE.open("r", encoding="utf-8") as f:
        _prompt_data = json.load(f)

    return _prompt_data

def _load_config() -> dict[str, Any]:
    """Loads the shared Qwen config once."""
    global _config_data

    if _config_data is not None:
        return _config_data

    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        _config_data = yaml.safe_load(f)

    return _config_data

def _load_model_and_tokenizer():
    """Loads model and tokenizer once through the shared wrapper."""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    config = _load_config()
    _model, _tokenizer = load_qwen_model(config)
    _model.eval()

    return _model, _tokenizer

def _build_examples(examples: list[dict[str, str]]) -> str:
    """Format few-shot examples."""
    blocks = []

    for i, ex in enumerate(examples, start=1):
        block = (
            f"Example {i}\n"
            f"Question: {ex['question']}\n"
            f"Reasoning: {ex['reasoning']}\n"
            f"Final Answer: {ex['answer']}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def _build_prompt(user_prompt: str) -> str:
    """Combine instruction, examples, and benchmark prompt."""
    prompt_data = _load_prompt_data()

    instruction = prompt_data["instruction"].strip()
    examples = _build_examples(prompt_data["examples"])

    return (
        f"{instruction}\n\n"
        f"{examples}\n\n"
        f"Now solve this problem.\n\n"
        f"{user_prompt}\n\n"
        f"Reasoning:"
    )


def cot_3shot(user_prompt: str) -> str:
    """Runs 3-shot CoT inference, basically the main function of this file."""
    model, tokenizer = _load_model_and_tokenizer()
    config = _load_config()
    inference_config = config.get("inference", {})

    max_new_tokens = int(inference_config.get("max_new_tokens", 256))
    temperature = float(inference_config.get("temperature", 0.0))
    top_p = float(inference_config.get("top_p", 1.0))
    do_sample = bool(inference_config.get("do_sample", False))

    full_prompt = _build_prompt(user_prompt)

    messages = [
        {
            "role": "user",
            "content": full_prompt,
        }
    ]

    text_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text_prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return decoded.strip()
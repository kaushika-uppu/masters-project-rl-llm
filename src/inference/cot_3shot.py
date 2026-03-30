"""
Runs 3-shot Chain-of-Thought prompting with Qwen2.5-7B.
Loads the shared prompt prefix, appends the benchmark prompt,
runs inference, and returns the model output text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from src.models.qwen_wrapper import load_qwen_model

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "cot_3shot" / "general_prompts.json"

_model = None
_tokenizer = None
_prompt_data: Optional[dict[str, Any]] = None


def _load_prompt_data() -> dict[str, Any]:
    """Load prompt data once."""
    global _prompt_data

    if _prompt_data is not None:
        return _prompt_data

    with PROMPT_FILE.open("r", encoding="utf-8") as f:
        _prompt_data = json.load(f)

    return _prompt_data


def _load_model_and_tokenizer():
    """Load model and tokenizer once."""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    config = {
        "model": {
            "name": "Qwen/Qwen2.5-7B",
            "torch_dtype": "float16",
            "device_map": "auto",
        }
    }

    _model, _tokenizer = load_qwen_model(config)
    _model.eval()

    return _model, _tokenizer


def _build_prompt(user_prompt: str) -> str:
    """Combine shared prompt prefix with benchmark prompt."""
    prompt_data = _load_prompt_data()
    base_prompt = prompt_data["prompt"].strip()

    return f"{base_prompt}\n\n{user_prompt}\n<reasoning>\n"


def cot_3shot(user_prompt: str) -> str:
    """Run 3-shot CoT inference."""
    model, tokenizer = _load_model_and_tokenizer()
    full_prompt = _build_prompt(user_prompt)

    inputs = tokenizer(full_prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return decoded.strip()
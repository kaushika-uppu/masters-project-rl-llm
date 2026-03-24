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
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "256"))

# Prompt file for the current benchmark (riddlebench)
PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "cot_3shot" / "riddlebench.json"

# Cache model, tokenizer, and prompt data.
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForCausalLM] = None
_prompt_data: Optional[dict] = None


def _load_prompt_data() -> dict:
    """Load prompt config once."""
    global _prompt_data

    if _prompt_data is not None:
        return _prompt_data

    with PROMPT_FILE.open("r", encoding="utf-8") as f:
        _prompt_data = json.load(f)

    return _prompt_data


def _load_model() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load model and tokenizer once."""
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if _tokenizer.pad_token_id is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=model_dtype,
        device_map="auto",
    )
    _model.eval()

    return _tokenizer, _model


def _build_examples(examples: list[dict]) -> str:
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
    """Run 3-shot CoT inference."""
    tokenizer, model = _load_model()
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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return decoded.strip()
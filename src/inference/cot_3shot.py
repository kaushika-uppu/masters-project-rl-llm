"""
Runs 3-shot Chain-of-Thought prompting with Qwen2.5-7B-Instruct.
Loads fixed examples from a JSON file, builds the final prompt, runs inference,
and returns the model's generated answer text.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from src.models.qwen_wrapper import load_qwen_model

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "cot_3shot" / "general_prompts.json"

# If bitsandbytes is already working on the machine, change this to "4bit".
QUANTIZATION = "none"

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
SAMPLE_COUNT = 1
MAX_NEW_TOKENS = 512
VERIFY_TOKENS = 160

_model = None
_tokenizer = None
_prompt_data: Optional[dict[str, Any]] = None


class StopOnSequence(StoppingCriteria):
    """Stops once the chosen closing tag has been emitted."""

    def __init__(self, stop_ids: list[int]) -> None:
        self.stop_ids = stop_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        if not self.stop_ids or input_ids.shape[1] < len(self.stop_ids):
            return False

        tail = input_ids[0, -len(self.stop_ids):].tolist()
        return tail == self.stop_ids


def _load_prompt_data() -> dict[str, Any]:
    """Loads the shared reasoning prompt once."""
    global _prompt_data

    if _prompt_data is not None:
        return _prompt_data

    with PROMPT_FILE.open("r", encoding="utf-8") as f:
        _prompt_data = json.load(f)

    return _prompt_data


def _load_model_and_tokenizer():
    """Loads the model through the shared wrapper once."""
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    config = {
        "model": {
            "name": MODEL_NAME,
            "torch_dtype": "float16",
            "device_map": "auto",
            "quantization": QUANTIZATION,
        }
    }

    _model, _tokenizer = load_qwen_model(config)
    _model.eval()

    return _model, _tokenizer


def _build_prompt(user_prompt: str) -> str:
    """Combines the shared reasoning prompt with the benchmark prompt."""
    prompt_data = _load_prompt_data()
    base_prompt = prompt_data["prompt"].strip()

    return f"{base_prompt}\n\n{user_prompt.strip()}\n"


def _detect_tags(user_prompt: str) -> tuple[Optional[str], Optional[str]]:
    """Finds the output schema requested by the benchmark prompt."""
    if "</python>" in user_prompt:
        return "<python>", "</python>"

    if "</answer>" in user_prompt:
        return "<answer>", "</answer>"

    return None, None


def _extract_payload(output: str, user_prompt: str) -> str:
    """Pulls out the answer block the benchmark actually scores."""
    open_tag, close_tag = _detect_tags(user_prompt)

    if open_tag and close_tag:
        match = re.search(
            rf"{re.escape(open_tag)}\s*(.*?)\s*{re.escape(close_tag)}",
            output,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        return ""

    return output.strip()


def _normalize_vote_key(payload: str, user_prompt: str) -> str:
    """Normalizes answers for voting."""
    cleaned = re.sub(r"\s+", " ", payload).strip()

    if "</python>" in user_prompt:
        return cleaned

    cleaned_no_commas = cleaned.replace(",", "")
    number_match = re.fullmatch(r"-?\d+(?:\.\d+)?", cleaned_no_commas)
    if number_match:
        return cleaned_no_commas

    return cleaned.lower()


def _build_stopping_criteria(user_prompt: str, tokenizer) -> Optional[StoppingCriteriaList]:
    """Stops when the benchmark's closing tag is generated."""
    _, close_tag = _detect_tags(user_prompt)
    if not close_tag:
        return None

    stop_ids = tokenizer.encode(close_tag, add_special_tokens=False)
    if not stop_ids:
        return None

    return StoppingCriteriaList([StopOnSequence(stop_ids)])


def _generate_one(
    prompt_text: str,
    user_prompt: str,
    *,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    max_new_tokens: int,
) -> str:
    """Runs one generation pass."""
    model, tokenizer = _load_model_and_tokenizer()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful reasoning assistant. "
                "Think step by step, verify the result, "
                "and follow the output format requested in the user's problem exactly."
            ),
        },
        {
            "role": "user",
            "content": prompt_text,
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    stopping_criteria = _build_stopping_criteria(user_prompt, tokenizer)
    if stopping_criteria is not None:
        generate_kwargs["stopping_criteria"] = stopping_criteria

    if do_sample:
        generate_kwargs["temperature"] = temperature if temperature is not None else 0.6
        generate_kwargs["top_p"] = top_p if top_p is not None else 0.9

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return decoded.strip()


def _build_verifier_prompt(user_prompt: str, candidate_outputs: list[str]) -> str:
    """Builds one verification pass over the sampled candidates."""
    blocks = []
    for i, candidate in enumerate(candidate_outputs, start=1):
        blocks.append(f"Candidate {i}:\n{candidate}")

    joined_candidates = "\n\n".join(blocks)

    return (
        "You are verifying candidate answers for a reasoning problem.\n"
        "Check each candidate against every condition in the original problem.\n"
        "Reject any candidate that fails even one condition.\n\n"
        "Verification rules:\n"
        "- For sequence or symbol-pattern tasks, check every position or component, not just one visible pattern.\n"
        "- Verify the answer against both the left side and the right side of the missing term.\n"
        "- Do not accept a candidate just because part of the pattern fits.\n"
        "- If all candidates are wrong, solve the problem yourself carefully.\n"
        "- Return the final result in exactly the format requested by the original problem.\n\n"
        f"Original problem:\n{user_prompt.strip()}\n\n"
        f"{joined_candidates}\n"
    )


def cot_3shot(user_prompt: str) -> str:
    """Runs pure shared CoT prompting with a single generation pass."""
    full_prompt = _build_prompt(user_prompt)

    return _generate_one(
        full_prompt,
        user_prompt,
        do_sample=False,
        temperature=None,
        top_p=None,
        max_new_tokens=MAX_NEW_TOKENS,
    )

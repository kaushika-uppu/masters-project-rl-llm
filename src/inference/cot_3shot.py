"""
Runs strict 3-shot Chain-of-Thought prompting with Qwen2.5-32B-Instruct.
Loads the shared prompt once, keeps benchmark questions intact, runs one
greedy batched generation pass, and trims echoed text so outputs stay in
the required <reasoning>/<answer> format without changing inference speed.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

from src.models.qwen_wrapper import load_qwen_model

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "cot_3shot" / "general_prompts.json"

QUANTIZATION = "4bit"
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
MAX_NEW_TOKENS = 1024
BATCH_SIZE = 8

_model = None
_tokenizer = None
_prompt_data: Optional[dict[str, Any]] = None


class StopOnAnswerTag(StoppingCriteria):
    def __init__(self, stop_token_ids: list[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        stop_len = len(self.stop_token_ids)

        if input_ids.shape[1] < stop_len:
            return False

        for row in input_ids:
            if row[-stop_len:].tolist() != self.stop_token_ids:
                return False

        return True


def _load_prompt_data() -> dict[str, Any]:
    global _prompt_data

    if _prompt_data is not None:
        return _prompt_data

    with PROMPT_FILE.open("r", encoding="utf-8") as f:
        _prompt_data = json.load(f)

    return _prompt_data


def _load_model_and_tokenizer():
    global _model, _tokenizer

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    config = {
        "model": {
            "name": MODEL_NAME,
            "torch_dtype": "float16",
            "device_map": "balanced",
            "quantization": QUANTIZATION,
        }
    }

    _model, _tokenizer = load_qwen_model(config)

    _tokenizer.padding_side = "left"
    _model.eval()
    _model.generation_config.do_sample = False
    _model.generation_config.temperature = None
    _model.generation_config.top_p = None
    _model.generation_config.top_k = None

    return _model, _tokenizer


def _strip_placeholder_text(user_prompt: str) -> str:
    text = user_prompt.strip()

    text = re.sub(
        r"(<reasoning>\s*)\[[^\]]*?\](\s*</reasoning>)",
        r"\1\2",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = re.sub(
        r"(<answer>\s*)\[[^\]]*?\](\s*</answer>)",
        r"\1\2",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    text = re.sub(
        r"Please provide your response.*?<answer>\s*\[.*?\]\s*</answer>\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return text.strip()


def _build_prompt(user_prompt: str) -> str:
    prompt_data = _load_prompt_data()
    base_prompt = prompt_data["prompt"].strip()
    cleaned_user_prompt = _strip_placeholder_text(user_prompt)
    return f"{base_prompt}\n\n{cleaned_user_prompt}\n"


def _minimal_format_cleanup(output: str) -> str:
    text = output.strip()
    text = re.sub(r"^\s*assistant\s*", "", text, flags=re.IGNORECASE).strip()

    first_reasoning = text.find("<reasoning>")
    first_answer = text.find("<answer>")
    tag_positions = [pos for pos in (first_reasoning, first_answer) if pos != -1]

    if tag_positions:
        text = text[min(tag_positions):].strip()

    return text


def _build_messages(prompt_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return your response in exactly this format:\n"
                "<reasoning>\n...\n</reasoning>\n"
                "<answer>\n...\n</answer>\n"
                "Do not add any extra text outside these tags."
            ),
        },
        {
            "role": "user",
            "content": prompt_text,
        },
    ]


def _generate_batch(
    prompt_texts: list[str],
    *,
    max_new_tokens: int,
) -> list[str]:
    model, tokenizer = _load_model_and_tokenizer()

    stop_token_ids = tokenizer.encode("</answer>", add_special_tokens=False)
    stopping_criteria = StoppingCriteriaList([StopOnAnswerTag(stop_token_ids)])

    messages_list = [_build_messages(prompt_text) for prompt_text in prompt_texts]

    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_list
    ]

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    prompt_length = inputs["input_ids"].shape[1]

    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "stopping_criteria": stopping_criteria,
    }

    with torch.no_grad():
        outputs = model.generate(**inputs, **generate_kwargs)

    decoded_outputs: list[str] = []
    for i in range(outputs.shape[0]):
        generated_ids = outputs[i, prompt_length:]
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        decoded_outputs.append(_minimal_format_cleanup(decoded_text))

    return decoded_outputs


def cot_3shot(user_prompt: str | list[str]) -> str | list[str]:
    if isinstance(user_prompt, list):
        all_outputs: list[str] = []
        total_start = time.time()

        for start in range(0, len(user_prompt), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(user_prompt))
            batch_num = start // BATCH_SIZE + 1
            batch_start = time.time()

            print(
                f"Processing batch {batch_num}: prompts {start + 1}-{end} of {len(user_prompt)}",
                flush=True,
            )

            prompt_chunk = user_prompt[start:end]
            full_prompts = [_build_prompt(prompt) for prompt in prompt_chunk]

            chunk_outputs = _generate_batch(
                full_prompts,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            all_outputs.extend(chunk_outputs)

            batch_elapsed = time.time() - batch_start
            total_elapsed = time.time() - total_start
            print(
                f"Finished batch {batch_num} in {batch_elapsed:.1f}s | total elapsed: {total_elapsed/60:.1f} min",
                flush=True,
            )

        total_elapsed = time.time() - total_start
        print(f"Total inference time: {total_elapsed/60:.2f} minutes", flush=True)
        return all_outputs

    full_prompt = _build_prompt(user_prompt)
    return _generate_batch(
        [full_prompt],
        max_new_tokens=MAX_NEW_TOKENS,
    )[0]


cot_3shot.is_batch = True

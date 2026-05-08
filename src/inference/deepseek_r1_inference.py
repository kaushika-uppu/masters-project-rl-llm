# Runs DeepSeek-R1-Distill-Qwen baseline evaluation.

from __future__ import annotations

import re
import yaml
import torch
from typing import Any

from src.inference.base_inference import BaseInference
from src.models.qwen_wrapper import load_qwen_model


class DeepSeekR1(BaseInference):
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.tokenizer.padding_side = "left"
        self.model.eval()

    def generate(self, prompts: list) -> list:
        final_results = []
        batch_size = self.config.get("inference", {}).get("batch_size", 8)

        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            prompt_batch = prompts[start:end]

            formatted_prompts = [
                (
                    "Solve with concise step-by-step reasoning.\n"
                    "After writing \\boxed{your_answer}, STOP immediately. Do not explain further.\n"
                    "End with exactly one final answer in this format:\n"
                    "\\boxed{your_answer}\n\n"
                    f"{p}"
                )
                for p in prompt_batch
            ]

            inputs = self.tokenizer(
                formatted_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.model.device)

            prompt_length = inputs["input_ids"].shape[1]
            inference_config = self.config.get("inference", {})
            do_sample = inference_config.get("do_sample", True)

            generate_kwargs = {
                "max_new_tokens": inference_config.get("max_new_tokens", 8192),
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }

            if do_sample:
                generate_kwargs["temperature"] = inference_config.get("temperature", 0.2)
                generate_kwargs["top_p"] = inference_config.get("top_p", 0.9)
                generate_kwargs["top_k"] = inference_config.get("top_k", 30)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generate_kwargs)

            for i in range(outputs.shape[0]):
                raw_text = self.tokenizer.decode(
                    outputs[i, prompt_length:],
                    skip_special_tokens=True,
                ).strip()

                boxed_matches = re.findall(r"\\boxed\{([^{}]*)\}", raw_text)
                
                if boxed_matches:
                    raw_text = raw_text.strip()

                final_results.append(raw_text)

        return final_results


def deepseek_r1():
    print("Loading DeepSeek-R1 configuration...")

    with open("configs/deepseek_r1.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model, tokenizer = load_qwen_model(config)

    deepseek_instance = DeepSeekR1(model=model, tokenizer=tokenizer, config=config)

    def batch_inference_fn(prompts):
        return deepseek_instance.generate(prompts)

    batch_inference_fn.is_batch = True
    return batch_inference_fn
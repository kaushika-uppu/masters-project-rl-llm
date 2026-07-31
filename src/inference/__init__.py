"""Inference and prompting logic."""

from src.inference.test_inference import dummy_inference
from src.inference.base_inference import BaseInference
from src.inference.cot_3shot import cot_3shot
from src.inference.limo_inference import limo
from src.inference.deepseek_r1_inference import deepseek_r1
from src.inference.sft_inference import sft_merged
from src.inference.qwen_base_inference import qwen_base

__all__ = ["dummy_inference", "BaseInference", "cot_3shot", "limo", "deepseek_r1", "sft_merged", "qwen_base", "rl_sft_merged"]

"""Inference and prompting logic."""

from src.inference.test_inference import dummy_inference
from src.inference.base_inference import BaseInference
from src.inference.cot_3shot import cot_3shot
from src.inference.limo_inference import LIMO

__all__ = ["dummy_inference", "BaseInference", "cot_3shot", "limo"]

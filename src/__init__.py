"""Main source code package for LLM training and inference."""

from src.function_registry import get_inference_function, get_available_functions

__all__ = ["get_inference_function", "get_available_functions"]
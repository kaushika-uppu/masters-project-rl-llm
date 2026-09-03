# src/function_registry.py
# keeps track of all functions that can be used for inference

from typing import Callable, Dict, List

from src.inference.model_configs import TRAINING_INFERENCE_CONFIGS

# Lazy imports - only import when functions are actually requested
def _get_registry() -> Dict[str, Callable[[], Callable[[str], str]]]:
    """Lazy load inference-function factories to avoid importing heavy dependencies.

    Every value is a zero-arg callable that returns the actual
    Callable[[str], str] generate function for that name.
    """
    from src.inference import (
        dummy_inference,
        cot_3shot,
        limo,
        deepseek_r1,
        sft_merged,
        qwen_base,
        load_training_inference,
    )

    registry: Dict[str, Callable[[], Callable[[str], str]]] = {
        "dummy": lambda: dummy_inference,
        "cot_3shot": lambda: cot_3shot,
        "limo": limo,
        "deepseek_r1": deepseek_r1,
        "sft_merged": sft_merged,
        "qwen_base": qwen_base,
    }
    for name, config in TRAINING_INFERENCE_CONFIGS.items():
        registry[name] = lambda config=config: load_training_inference(**config)
    return registry


def get_available_functions() -> List[str]:
    """Return list of available inference function names."""
    return list(_get_registry().keys())


def get_inference_function(name: str) -> Callable[[str], str]:
    """Get an inference function by name. Uses lazy imports to avoid loading heavy dependencies."""
    registry = _get_registry()
    if name not in registry:
        raise ValueError(f"Unknown inference function '{name}'. Available: {list(registry.keys())}")
    return registry[name]()

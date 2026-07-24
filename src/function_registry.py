# src/function_registry.py
# keeps track of all functions that can be used for inference

from typing import Dict, Callable

# Lazy imports - only import when functions are actually requested
def _get_registry() -> Dict[str, Callable[[str], str]]:
    """Lazy load inference functions to avoid importing heavy dependencies."""
    from src.inference import dummy_inference, cot_3shot
    from src.inference.limo_inference import limo
    from src.inference.deepseek_r1_inference import deepseek_r1
    from src.inference.rl_sft_inference import rl_sft_merged

    return {
        "dummy": dummy_inference,
        "cot_3shot": cot_3shot,
        "limo": limo,
        "deepseek_r1": deepseek_r1,
        "rl_sft_merged": rl_sft_merged
    }

def get_available_functions() -> list[str]:
    """Return list of available inference function names."""
    return ["dummy", "cot_3shot", "limo", "deepseek_r1", "rl_sft_merged"]

def get_inference_function(name: str) -> Callable[[str], str]:
    """Get an inference function by name. Uses lazy imports to avoid loading heavy dependencies."""
    available = get_available_functions()
    if name not in available:
        raise ValueError(f"Unknown inference function '{name}'. Available: {available}")

    registry = _get_registry()
    return registry[name]

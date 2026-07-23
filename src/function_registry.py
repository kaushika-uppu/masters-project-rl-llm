# src/function_registry.py
# keeps track of all functions that can be used for inference

from typing import Dict, Callable

# Lazy imports - only import when functions are actually requested
def _get_registry() -> Dict[str, Callable[[str], str]]:
    """Lazy load inference functions to avoid importing heavy dependencies."""
    from src.inference import dummy_inference, cot_3shot
    from src.inference.limo_inference import limo
    from src.inference.deepseek_r1_inference import deepseek_r1
    from src.inference.sft_inference import sft_merged
    from src.inference.qwen_base_inference import qwen_base

    return {
        "dummy": dummy_inference,
        "cot_3shot": cot_3shot,
        "limo": limo,
        "deepseek_r1": deepseek_r1,
        "sft_merged": sft_merged,
        "qwen_base": qwen_base
    }

def get_available_functions() -> list[str]:
    """Return list of available inference function names."""
    return ["dummy", "cot_3shot", "limo", "deepseek_r1", "sft_merged", "qwen_base"]

def get_inference_function(name: str) -> Callable[[str], str]:
    """Get an inference function by name. Uses lazy imports to avoid loading heavy dependencies."""
    available = get_available_functions()
    if name not in available:
        raise ValueError(f"Unknown inference function '{name}'. Available: {available}")

    # Lazy load the registry
    registry = _get_registry()

    if name == 'limo':
        from src.inference.limo_inference import limo
        return limo()

    if name == 'deepseek_r1':
        from src.inference.deepseek_r1_inference import deepseek_r1
        return deepseek_r1()

    if name == 'sft_merged':
        from src.inference.sft_inference import sft_merged
        return sft_merged()

    if name == 'qwen_base':
        from src.inference.qwen_base_inference import qwen_base
        return qwen_base()

    return registry[name]

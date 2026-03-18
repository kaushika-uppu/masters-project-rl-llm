# src/function_registry.py
# keeps track of all functions that can be used for inference

from typing import Dict, Callable
from src.inference import dummy_inference

INFER_FUNCTION_REGISTRY: Dict[str, Callable[[str], str]] = {
    "dummy": dummy_inference,
}

def get_available_functions() -> list[str]:
    """Return list of available inference function names."""
    return list(INFER_FUNCTION_REGISTRY.keys())

def get_inference_function(name: str) -> Callable[[str], str]:
    if name not in INFER_FUNCTION_REGISTRY:
        available = get_available_functions()
        raise ValueError(f"Unknown inference function '{name}'. Available: {available}")
    
    return INFER_FUNCTION_REGISTRY[name]

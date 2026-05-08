# src/function_registry.py
# keeps track of all functions that can be used for inference

from typing import Dict, Callable
from src.inference import dummy_inference, cot_3shot
from src.inference.limo_inference import limo
from src.inference.deepseek_r1_inference import deepseek_r1

INFER_FUNCTION_REGISTRY: Dict[str, Callable[[str], str]] = {
    "dummy": dummy_inference,
    "cot_3shot": cot_3shot,
    "limo": limo,
    "deepseek_r1": deepseek_r1
}

def get_available_functions() -> list[str]:
    """Return list of available inference function names."""
    return list(INFER_FUNCTION_REGISTRY.keys())

def get_inference_function(name: str) -> Callable[[str], str]:
    if name not in INFER_FUNCTION_REGISTRY:
        available = get_available_functions()
        raise ValueError(f"Unknown inference function '{name}'. Available: {available}")
    if name == 'limo':
        return limo()
        
    if name == 'deepseek_r1':
        return deepseek_r1()
    
    return INFER_FUNCTION_REGISTRY[name]

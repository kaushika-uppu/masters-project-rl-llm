"""Main source code package for LLM training and inference."""

# Lazy imports to avoid loading heavy dependencies (like vllm) unless needed
def get_inference_function(name: str):
    """Get an inference function by name."""
    from src.function_registry import get_inference_function as _get_func
    return _get_func(name)

def get_available_functions():
    """Get list of available inference function names."""
    from src.function_registry import get_available_functions as _get_funcs
    return _get_funcs()

__all__ = ["get_inference_function", "get_available_functions"]
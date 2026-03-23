"""Registry for loading and instantiating models."""

from typing import Dict, Any, Tuple, Callable, List
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.models.qwen_wrapper import load_qwen_model

ModelLoaderCallable = Callable[[Dict[str, Any]], Tuple[PreTrainedModel, PreTrainedTokenizerBase]]

MODEL_REGISTRY: Dict[str, ModelLoaderCallable] = {
    "qwen": load_qwen_model
}

def get_supported_models() -> List[str]:
    """Return a list of supported base model families."""
    return list(MODEL_REGISTRY.keys())

def get_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Route the config to the correct model loader."""
    model_name = config.get("model", {}).get("name", "").lower()
    
    for family_key, loader_func in MODEL_REGISTRY.items():
        if family_key in model_name:
            return loader_func(config)
            
    available = get_supported_models()
    raise ValueError(f"Unknown model family: '{model_name}'. Supported families: {available}")
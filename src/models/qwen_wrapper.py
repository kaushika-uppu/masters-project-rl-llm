from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
import torch
from typing import Dict, Any, Tuple

def load_qwen_model(config: Dict[str, Any]) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Loads the Qwen model and tokenizer based on the given config settings.

    Args:
        config (Dict[str, Any]): The configuration dictionary containing model parameters.
        
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizerBase]: The loaded model and tokenizer.
    """
    model_config = config.get('model', {})
    model_name = model_config.get('name', 'Qwen/Qwen2.5-7B')

    print(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype_str = model_config.get('torch_dtype', 'bfloat16')
    torch_dtype = getattr(torch, dtype_str)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=model_config.get('device_map', 'auto'),
        trust_remote_code=True
    )
    
    return model, tokenizer
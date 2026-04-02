from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, BitsAndBytesConfig
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
    # use 7B-Instruct as default
    model_name = model_config.get('name', 'Qwen/Qwen2.5-7B-Instruct')

    print(f"Loading tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype_str = model_config.get('torch_dtype', 'bfloat16')
    torch_dtype = getattr(torch, dtype_str)

    quantization_config = None
    quant_type = model_config.get('quantization', 'none').lower()

    if quant_type == '4bit':
        print("Applying 4-bit NF4 quantization to save VRAM...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_compute_dtype = torch_dtype,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_use_double_quant = True
        )
    elif quant_type == "8bit":
        print("Applying standard 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=model_config.get('device_map', 'auto'),
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    
    return model, tokenizer
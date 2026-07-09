from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from peft import LoraConfig, get_peft_model, AutoPeftModelForCausalLM

import yaml
import argparse

from src.training.sft import run_sft
from src.training.rl.run_rl import run_rl

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    validate_config(config)
    return config

def validate_config(config: dict) -> None:
    training_modes = ["sft", "rl"]
    active_modes = [mode for mode in training_modes if mode in config]

    if len(active_modes) == 0:
        raise ValueError("No training mode specified. Available: sft, rl")

def load_model_and_tokenizer(model_config: dict) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load model and tokenizer from config."""
    # option 1: load from saved checkpoint
    if model_config.get("from_checkpoint"):
        checkpoint_path = model_config["from_checkpoint"]
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Check if it's a PEFT model
        if model_config.get('is_peft_checkpoint', False):
            model = AutoPeftModelForCausalLM.from_pretrained(
                checkpoint_path,
                is_trainable=True,
                torch_dtype=model_config.get('torch_dtype', 'auto'),
                device_map=model_config.get('device_map', 'auto')
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=model_config.get('torch_dtype', 'auto'),
                device_map=model_config.get('device_map', 'auto')
            )
        
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # option 2: load from model name/path
    elif model_config.get('name'):
        model_name = model_config['name']
        print(f"Loading model: {model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_config.get('torch_dtype', 'auto'),
            device_map=model_config.get('device_map', 'auto'),
            **model_config.get('additional_kwargs', {})
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # option 3: load from saved config
    elif model_config.get('from_config'):
        config_path = model_config['from_config']
        print(f"Loading model config from: {config_path}")
        
        with open(config_path, 'r') as f:
            saved_config = yaml.safe_load(f)
        
        model_name = saved_config['model']['name']
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=saved_config['model'].get('torch_dtype', 'auto'),
            device_map=saved_config['model'].get('device_map', 'auto')
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    else:
        raise ValueError("Must specify either 'name', 'from_checkpoint', or 'from_config' in model config")
    
    return model, tokenizer

def apply_lora(model: PreTrainedModel, lora_config: dict) -> PreTrainedModel:
    """Apply LoRA to model if specified in config."""
    print("Applying LoRA configuration...")
    peft_config = LoraConfig(
        r=lora_config.get('r', 16),
        lora_alpha=lora_config.get('alpha', 32),
        target_modules=lora_config.get('target_modules', ["q_proj", "v_proj"]),
        lora_dropout=lora_config.get('dropout', 0.05),
        bias=lora_config.get('bias', 'none'),
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model

def run_training(config: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    """Run the training pipeline."""
    # Run SFT if present in config
    if 'sft' in config:
        print("\n" + "="*50)
        print("Starting SFT Training")
        print("="*50)

        sft_config = config['sft']
        run_sft(
            model=model,
            tokenizer=tokenizer,
            dataset=sft_config['dataset'],
            output_dir=sft_config['output_dir'],
            sft_config=sft_config
        )

        print(f"SFT complete! Model saved to {sft_config['output_dir']}")

        # If RL is also present, reload the model from SFT checkpoint
        if 'rl' in config:
            print(f"\nReloading model from SFT checkpoint for RL training...")
            upd_config = {
                "from_checkpoint": sft_config['output_dir'],
                "torch_dtype": config['model'].get('torch_dtype', 'auto'),
                "device_map": config['model'].get('device_map', 'auto')
            }
            if 'lora' in config:
                upd_config['is_peft_checkpoint'] = True
            model, tokenizer = load_model_and_tokenizer(upd_config)
    
    # Run RL if present in config
    if 'rl' in config:
        print("\n" + "="*50)
        print("Starting RL Training")
        print("="*50)

        run_rl(config, model, tokenizer)
        print(f"RL complete! Model saved to {config['rl']['output_dir']}")

def main():
    parser = argparse.ArgumentParser(description="Run project training pipeline from config.")
    parser.add_argument("config_path", type=str, help="Path to the config YAML file.")
    args = parser.parse_args()

    config = load_config(args.config_path)

    model_config = config.get("model", {})
    model, tokenizer = load_model_and_tokenizer(model_config)

    # Set padding token if not set (required for SFTTrainer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA if present in config and not loading a PEFT checkpoint
    if 'lora' in config and not model_config.get('is_peft_checkpoint', False):
        lora_config = config['lora']
        model = apply_lora(model, lora_config)

    run_training(config, model, tokenizer)

    print("\n" + "="*50)
    print("All training complete!")
    print("="*50)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args(): 
    parser = argparse.ArgumentParser(description="Evaluate LLMs on benchmarks")

    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to lora checkpoint"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output file for merged model"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = args.adapter_path
    output_path = args.output_path

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print("Merging LoRA weights...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    main()
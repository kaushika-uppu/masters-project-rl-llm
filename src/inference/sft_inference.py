import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any
from src.inference.base_inference import BaseInference

class SFTInference(BaseInference):
    
    def format_prompt(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful reasoning assistant. Break down your reasoning into clear, logical steps."},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate(self, prompt: str) -> str:
        chat_prompt = self.format_prompt(prompt)
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,  
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return generated_text

def sft_merged():
    model_path = "./checkpoints/sft_merged"
    print(f"Loading merged model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    config = {"model": {"is_instruct": True}}
    inference_engine = SFTInference(model=model, tokenizer=tokenizer, config=config)
    
    return inference_engine.generate
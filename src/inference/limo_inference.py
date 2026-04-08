import torch
from typing import Dict, Any, List
import re
import yaml
from src.inference.base_inference import BaseInference
from src.models.model_registry import get_model_and_tokenizer


class LIMO(BaseInference):
    def post_process_output(self, raw_output: str) -> str:
        """
        Extracts the answer from LIMO's \boxed{} format and appends it 
        inside <answer></answer> tags so benchmarks can parse it.
        """
        answer = ''
        if "\\boxed{" in raw_output:
            start_idx = raw_output.rfind("\\boxed{") + 7
            depth = 1
            
            # bracket matching algorithm to parse answer
            for i in range(start_idx, len(raw_output)):
                if raw_output[i] == '{':
                    depth += 1
                elif raw_output[i] == '}':
                    depth -= 1
                
                if depth == 0:
                    answer = raw_output[start_idx:i]
                    break
            if answer:
                # return raw output to keep reasoning, append final answer within <answer> tags for evaluation
                return raw_output + f"\n\n<answer>\n{answer.strip()}\n</answer>"
            
        return raw_output


    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": prompt}
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors='pt').to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get('inference', {}).get('max_new_tokens', 32768),
                temperature=self.config.get('inference', {}).get('temperature', 0.7),
                top_p=self.config.get('inference', {}).get('top_p', 0.95),
                do_sample=self.config.get('inference', {}).get('do_sample', True),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        input_length = inputs.input_ids.shape[1]
        generated_ids = outputs[0][input_length:]

        raw_output = self.tokenizer.decode(generated_ids,
                                           skip_special_tokens=True)
        
        return self.post_process_output(raw_output)
    
def limo():
    """
    Function that sets up the LIMO environment.
    Loads the config, initializes the model, creates LIMO object, 
    and returns the .generate method.
    """
    print("Loading LIMO configuration...")
    with open("configs/limo.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Loading model {config['model']['name']}...")
    model, tokenizer = get_model_and_tokenizer(config)
    
    limo_instance = LIMO(model, tokenizer, config)
    return limo_instance.generate
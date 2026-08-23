import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.inference.base_inference import BaseInference
from .constants import (
    RIDDLEBENCH_SYSTEM_PROMPT,
    DEEPTHEOREM_SYSTEM_PROMPT,
    COT_3SHOT_SYSTEM_PROMPT,
    COT_3SHOT_EXAMPLES
)

MODEL_PATH = "./checkpoints/rl_tree_base_complete"

SYSTEM_PROMPT = RIDDLEBENCH_SYSTEM_PROMPT
# SYSTEM_PROMPT = DEEPTHEOREM_SYSTEM_PROMPT
# SYSTEM_PROMPT = "You are a helpful assistant."

USE_COT_3SHOT = False  # Set to True to enable CoT 3-shot examples

class TrainingInference(BaseInference):

    def format_prompt(self, prompt: str) -> str:
        # For CoT 3-shot, prepend examples to the user prompt
        if USE_COT_3SHOT:
            user_content = f"{COT_3SHOT_EXAMPLES}\n\n{prompt}"
        else:
            user_content = prompt

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
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

def rl_tree_base_checkpoint_350_merged():
    model_path = MODEL_PATH
    print(f"Loading merged model from {model_path}...")
    print(f"System prompt: {SYSTEM_PROMPT}")
    print(f"CoT 3-shot: {'ENABLED' if USE_COT_3SHOT else 'DISABLED'}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    config = {"model": {"is_instruct": True}}
    inference_engine = TrainingInference(model=model, tokenizer=tokenizer, config=config)

    return inference_engine.generate

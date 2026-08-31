import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference.base_inference import BaseInference
from src.inference.constants import COT_3SHOT_EXAMPLES


class TrainingInference(BaseInference):

    def __init__(self, model, tokenizer, config, system_prompt: str, use_cot_3shot: bool = False):
        super().__init__(model=model, tokenizer=tokenizer, config=config)
        self.system_prompt = system_prompt
        self.use_cot_3shot = use_cot_3shot

    def format_prompt(self, prompt: str) -> str:
        if self.use_cot_3shot:
            user_content = f"{COT_3SHOT_EXAMPLES}\n\n{prompt}"
        else:
            user_content = prompt

        messages = [
            {"role": "system", "content": self.system_prompt},
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


def load_training_inference(model_path: str, system_prompt: str, use_cot_3shot: bool = False):
    print(f"Loading merged model from {model_path}...")
    print(f"System prompt: {system_prompt}")
    print(f"CoT 3-shot: {'ENABLED' if use_cot_3shot else 'DISABLED'}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    config = {"model": {"is_instruct": True}}
    inference_engine = TrainingInference(
        model=model,
        tokenizer=tokenizer,
        config=config,
        system_prompt=system_prompt,
        use_cot_3shot=use_cot_3shot,
    )

    return inference_engine.generate

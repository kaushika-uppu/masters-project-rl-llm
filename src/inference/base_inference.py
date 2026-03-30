from abc import ABC, abstractmethod
from typing import Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizerBase

class BaseInference(ABC):
    """
    Base class for inference pipelines.
    """
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizerBase, 
        config: Dict[str, Any]
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.model.eval()

    def format_prompt(self, prompt: str) -> str:
        """
        Wraps the prompt in the model's specific chat template if it's an Instruct model. Otherwise, returns the raw prompt for base models.
        """
        is_instruct = self.config.get('model', {}).get('is_instruct', False)

        if is_instruct:
            messages = [
                {'role': 'system', 'content': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        return prompt
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Executes specific inference pipeline and returns the text."""
        pass
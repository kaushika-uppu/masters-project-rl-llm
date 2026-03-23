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

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Executes specific inference pipeline and returns the text."""
        pass
"""State-summary embedder for semantic node merging.

Uses a small sentence-embedding model loaded in-process (mean-pooled, L2-normalized) —
no extra pip dependency beyond transformers/torch. `encode(list[str]) -> list[vec]` is
the interface StateMatcher expects as `embed_fn`. torch/transformers import lazily so
this module stays importable in the sandbox.
"""

from __future__ import annotations

from typing import Optional


class TransformersEmbedder:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None
        self._tok = None

    def _lazy(self):
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            if self.device:
                self._model.to(self.device)
            self._model.eval()

    def encode(self, texts: list[str]) -> list[list[float]]:
        import torch

        self._lazy()
        enc = self._tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            hidden = self._model(**enc).last_hidden_state  # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, T, 1)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
        return emb.cpu().float().tolist()

    __call__ = encode


def build_state_matcher(merge_cfg: Optional[dict] = None):
    """Build a StateMatcher from the rl.merge config.

    merge_cfg keys: semantic (bool, default True), embed_model (str),
    cosine_threshold (float), device (str). semantic=False → exact-key matching.
    """
    from .merge import StateMatcher

    cfg = merge_cfg or {}
    if not cfg.get("semantic", True):
        return StateMatcher()  # exact-key fallback
    embedder = TransformersEmbedder(
        model_name=cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
        device=cfg.get("device"),
    )
    return StateMatcher(
        embed_fn=embedder.encode, cosine_threshold=cfg.get("cosine_threshold", 0.9)
    )

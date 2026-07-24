"""In-process Policy/Judge (no servers, no HTTP) — matches how SFT/eval already run.

- TransformersPolicy generates steps from a model you already hold (typically the model
  being trained -> on-policy, no weight-sync).
- TransformersJudge runs a judge model loaded in-process.

Implements the same rl.types.Policy/Judge protocols as the served versions, so the RL
engine is unchanged. torch/transformers are imported lazily (only at call time).
"""
from __future__ import annotations

from typing import Optional

from src.judge.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from src.judge.step_judge import _extract_json
from .policy import _extract_step, build_step_messages
from .types import Problem, StepJudgement


def _generate(model, tokenizer, messages: list[dict], n: int, temperature: float, max_new_tokens: int) -> list[str]:
    import torch
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    do_sample = temperature and temperature > 0
    with torch.no_grad():
        out = model.generate(
            **enc, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            num_return_sequences=n, max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen = out[:, enc["input_ids"].shape[1]:]
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]


def _generate_batch(model, tokenizer, batch_messages: list, temperature: float, max_new_tokens: int) -> list[str]:
    """Generate ONE continuation for each message-set in the batch, in a single forward.

    Left-pads so the prompt occupies the same column count for every row; the generated
    tokens are then the same suffix slice for all rows. This is the throughput win: the
    whole GRPO group (and any retries) go through the model together instead of serially.
    """
    import torch
    if not batch_messages:
        return []
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch_messages]
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    finally:
        tokenizer.padding_side = prev_side
    do_sample = bool(temperature and temperature > 0)
    with torch.no_grad():
        out = model.generate(
            **enc, do_sample=do_sample,
            temperature=temperature if do_sample else None,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen = out[:, enc["input_ids"].shape[1]:]
    return [tokenizer.decode(g, skip_special_tokens=True) for g in gen]


class TransformersPolicy:
    def __init__(self, model, tokenizer, temperature: float = 0.8, max_new_tokens: int = 384):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def _messages(self, problem: Problem, history: list[str], extra_user: str = "") -> list[dict]:
        return build_step_messages(problem, history, extra_user)

    def propose_steps(self, problem: Problem, history: list[str], k: int) -> list[str]:
        outs = _generate(self.model, self.tokenizer, self._messages(problem, history),
                         n=k, temperature=self.temperature, max_new_tokens=self.max_new_tokens)
        return [_extract_step(o) for o in outs]

    def revise_step(self, problem: Problem, history: list[str], failed_step: str, reason: str) -> str:
        return self.revise_steps_batch(problem, [(history, failed_step, reason)])[0]

    # -- batched (used by RolloutEngine lock-step expansion) ---------------------
    @staticmethod
    def _revise_extra(failed_step: str, reason: str) -> str:
        return (f"Your previous step was rejected.\nRejected step: {failed_step}\n"
                f"Reason: {reason}\nProduce a corrected, valid next step instead.")

    def propose_steps_batch(self, problem: Problem, histories: list[list[str]]) -> list[str]:
        """One next step per history, generated in a single batched forward."""
        batch = [self._messages(problem, h) for h in histories]
        outs = _generate_batch(self.model, self.tokenizer, batch,
                               temperature=self.temperature, max_new_tokens=self.max_new_tokens)
        return [_extract_step(o) for o in outs]

    def revise_steps_batch(self, problem: Problem, items: list) -> list[str]:
        """items: (history, failed_step, reason). One revised step per item, batched."""
        batch = [self._messages(problem, h, self._revise_extra(fs, r)) for (h, fs, r) in items]
        outs = _generate_batch(self.model, self.tokenizer, batch,
                               temperature=self.temperature, max_new_tokens=self.max_new_tokens)
        return [_extract_step(o) for o in outs]


class TransformersJudge:
    def __init__(self, model, tokenizer, max_new_tokens: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def judge_step(self, problem: Problem, history: list[str], step: str) -> StepJudgement:
        return self.judge_steps_batch(problem, [(history, step)])[0]

    def judge_steps_batch(self, problem: Problem, items: list) -> list[StepJudgement]:
        """items: (history, step). All steps judged in a single batched forward."""
        batch = [
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_user_prompt(problem.statement, h, s)},
            ]
            for (h, s) in items
        ]
        raws = _generate_batch(self.model, self.tokenizer, batch,
                               temperature=0.0, max_new_tokens=self.max_new_tokens)
        return [self._parse(raw, h, s) for raw, (h, s) in zip(raws, items)]

    @staticmethod
    def _parse(raw: str, history: list[str], step: str) -> StepJudgement:
        obj = _extract_json(raw)
        if obj is None:  # conservative: unparseable judge output => invalid, no state advance
            return StepJudgement(valid=False, reason="judge parse error",
                                 state_summary="\n".join(history))
        verdict = obj.get("verdict")
        verdict = verdict.upper() if isinstance(verdict, str) and verdict.upper() in ("PROVED", "DISPROVED") else None
        return StepJudgement(
            valid=bool(obj.get("valid", False)),
            reason=str(obj.get("reason", "")),
            state_summary=str(obj.get("state_summary", "") or "\n".join(history + [step])),
            is_terminal=bool(obj.get("is_terminal", False)),
            verdict=verdict,
        )

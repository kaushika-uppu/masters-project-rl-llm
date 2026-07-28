"""Served generation policy (implements rl.types.Policy).

Generates ONE proof step at a time from an OpenAI-compatible endpoint (vLLM). Used for
tree rollouts. The model being *trained* is separate (HF model in the trainer); this is
the fast sampler. Keep weights in sync between runs (trainer reloads / vLLM restarts).
"""

from __future__ import annotations

import re

from src.data.deeptheorem import PROVE_OR_DISPROVE_SYSTEM_PROMPT
from src.judge.step_judge import OpenAICompatClient

from .types import Problem

_STEP_RE = re.compile(r"<step>(.*?)</step>", re.DOTALL)
_VERDICT_RE = re.compile(r"^\s*verdict\s*:?", re.IGNORECASE)

# The exact instruction the policy is generation-prompted with. Shared by the served and
# in-process policies AND by the trainer, so the text the model is scored on (sample_builder
# -> grpo_tree._seq_logprob) matches the text it actually generated under. Do not fork this.
STEP_INSTRUCTION = (
    "Continue with EXACTLY ONE next step wrapped in <step>...</step>. "
    "If the argument is complete, instead output a final line: "
    "'Verdict: PROVED' or 'Verdict: DISPROVED'."
)


def _format_history(history: list[str]) -> str:
    return "\n".join(f"<step>{h}</step>" for h in history)


def build_step_messages(
    problem: Problem, history: list[str], extra_user: str = ""
) -> list[dict]:
    """The single source of truth for the policy's per-step generation prompt.

    Used to GENERATE (VLLMPolicy / TransformersPolicy) and to SCORE (sample_builder ->
    trainer), so the training log-prob is over exactly the rendering the model saw.
    """
    msgs = [
        {"role": "system", "content": PROVE_OR_DISPROVE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Prove or disprove the following:\n{problem.statement}",
        },
    ]
    if history:
        msgs.append({"role": "assistant", "content": _format_history(history)})
    instr = (extra_user + "\n" + STEP_INSTRUCTION) if extra_user else STEP_INSTRUCTION
    msgs.append({"role": "user", "content": instr})
    return msgs


def reconstruct_continuation(step: str) -> str:
    """Rebuild the assistant text the policy emitted for a stored step. Normal steps were
    generated wrapped in <step>...</step>; terminal steps were a bare 'Verdict: ...' line."""
    if _VERDICT_RE.match(step):
        return step.strip()
    return f"<step>{step.strip()}</step>"


def _extract_step(text: str) -> str:
    m = _STEP_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()  # may be a "Verdict: ..." line; judge will detect terminal


class VLLMPolicy:
    def __init__(
        self,
        client: OpenAICompatClient,
        temperature: float = 0.8,
        max_tokens: int = 384,
    ):
        self.client = client
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _messages(
        self, problem: Problem, history: list[str], extra_user: str = ""
    ) -> list[dict]:
        return build_step_messages(problem, history, extra_user)

    def propose_steps(self, problem: Problem, history: list[str], k: int) -> list[str]:
        outs = self.client.chat(
            self._messages(problem, history),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=k,
            stop=["</step>"],
        )
        return [_extract_step(o) for o in outs]

    def revise_step(
        self, problem: Problem, history: list[str], failed_step: str, reason: str
    ) -> str:
        extra = (
            f"Your previous step was rejected.\nRejected step: {failed_step}\n"
            f"Reason: {reason}\nProduce a corrected, valid next step instead."
        )
        outs = self.client.chat(
            self._messages(problem, history, extra_user=extra),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,
            stop=["</step>"],
        )
        return _extract_step(outs[0])

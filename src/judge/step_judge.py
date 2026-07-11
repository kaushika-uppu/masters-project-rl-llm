"""Helpers for parsing step judge output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from src.judge.prompts import JUDGE_SYSTEM_PROMPT, build_judge_user_prompt
from src.training.rl.types import Problem, StepJudgement


def _extract_json(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None

    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _normalize_verdict(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in {"PROVED", "DISPROVED"} else None


def judgement_from_obj(
    obj: dict[str, Any] | None,
    *,
    fallback_history: list[str] | None = None,
    fallback_step: str = "",
) -> StepJudgement:
    if obj is None:
        return StepJudgement(
            valid=False,
            reason="judge parse error",
            state_summary="\n".join(fallback_history or []),
        )

    hist = fallback_history or []
    state = str(obj.get("state_summary") or "\n".join(hist + ([fallback_step] if fallback_step else [])))
    verdict = _normalize_verdict(obj.get("verdict"))
    is_terminal = bool(obj.get("is_terminal", False)) or verdict is not None
    valid = bool(obj.get("valid", False))
    reason = str(obj.get("reason", ""))
    if is_terminal and verdict is None:
        valid = False
        reason = f"{reason}; invalid terminal verdict".strip("; ")
    return StepJudgement(
        valid=valid,
        reason=reason,
        state_summary=state,
        is_terminal=is_terminal,
        verdict=verdict,
    )


def parse_step_judgement(raw: str, *, history: list[str] | None = None, step: str = "") -> StepJudgement:
    return judgement_from_obj(_extract_json(raw), fallback_history=history, fallback_step=step)


@dataclass
class OpenAICompatClient:
    complete_fn: Callable[[list[dict[str, str]]], str] | None = None

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self.complete_fn is None:
            raise RuntimeError("OpenAICompatClient requires a complete_fn for now.")
        return self.complete_fn(messages)


class LLMJudge:
    def __init__(self, client: OpenAICompatClient):
        self.client = client

    def judge_step(self, problem: Problem, history: list[str], step: str) -> StepJudgement:
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_prompt(problem.statement, history, step)},
        ]
        return parse_step_judgement(self.client.complete(messages), history=history, step=step)

    def judge_steps_batch(self, problem: Problem, items: list[tuple[list[str], str]]) -> list[StepJudgement]:
        return [self.judge_step(problem, history, step) for history, step in items]


class DeterministicJudge:
    def judge_step(self, problem: Problem, history: list[str], step: str) -> StepJudgement:
        text = step.strip()
        low = text.lower()
        if any(marker in low for marker in ("invalid", "bad step", "nonsense")):
            return StepJudgement(
                valid=False,
                reason="deterministic invalid marker",
                state_summary="\n".join(history),
            )
        verdict = None
        is_terminal = False
        if "verdict" in low:
            if "disproved" in low:
                verdict = "DISPROVED"
            elif "proved" in low:
                verdict = "PROVED"
            is_terminal = verdict is not None
        return StepJudgement(
            valid=True,
            reason="deterministic accepted",
            state_summary="\n".join(history + [text]),
            is_terminal=is_terminal,
            verdict=verdict,
        )

    def judge_steps_batch(self, problem: Problem, items: list[tuple[list[str], str]]) -> list[StepJudgement]:
        return [self.judge_step(problem, history, step) for history, step in items]

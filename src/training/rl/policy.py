"""Prompt helpers for the step-level RL policy."""

from __future__ import annotations

import re

from .types import Problem


SYSTEM_PROMPT = (
    "You are an expert mathematical reasoning assistant. Prove or disprove the "
    "claim step by step. Return one useful next step at a time."
)

_STEP_RE = re.compile(r"<step>(.*?)</step>", re.IGNORECASE | re.DOTALL)
_PREFIX_RE = re.compile(r"^\s*(?:[-*]|\d+[.)]|step\s+\d+\s*:)\s*", re.IGNORECASE)


def _strip(text: str) -> str:
    return text.strip().strip("\"'")


def _extract_step(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    match = _STEP_RE.search(text)
    if match:
        return _strip(match.group(1))

    for line in text.splitlines():
        line = _PREFIX_RE.sub("", line).strip()
        if line:
            return _strip(line)
    return _strip(text)


def reconstruct_continuation(step: str) -> str:
    step = (step or "").strip()
    if step.upper().startswith("VERDICT"):
        return step
    return f"<step>{step}</step>"


def build_step_messages(
    problem: Problem,
    history: list[str],
    extra_user: str = "",
) -> list[dict[str, str]]:
    prior = "\n".join(f"<step>{s}</step>" for s in history)
    if not prior:
        prior = "(no previous steps)"

    user = (
        f"Prove or disprove the following:\n{problem.statement}\n\n"
        f"Previous steps:\n{prior}\n\n"
        "Return exactly one next step in <step>...</step>. "
        "If the proof is complete, return Verdict: PROVED or Verdict: DISPROVED."
    )
    if extra_user:
        user = f"{user}\n\n{extra_user}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

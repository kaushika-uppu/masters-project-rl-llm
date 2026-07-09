"""Core data contracts for the RL proof pipeline.

The pipeline consumes a list of `Problem` (the curation/subsetting of which DeepTheorem
questions to use is handled elsewhere). Everything downstream depends only on these
types and the Policy/Judge protocols, so models and judges are swappable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class Problem:
    """A single prove-or-disprove instance handed to the pipeline."""

    id: str
    statement: str  # the "prove or disprove ..." claim
    label: bool  # ground-truth verdict: True = PROVED, False = DISPROVED
    reference: str = (
        ""  # worked reference (OFFLINE only — judge calibration, never live reward)
    )
    domain: str = ""
    difficulty: Optional[float] = None


@dataclass
class StepJudgement:
    """Judge output for one candidate step given the current state."""

    valid: bool  # is this inference sound given the state so far?
    reason: str = ""  # short rationale (for self-correction feedback on retry)
    state_summary: str = ""  # updated (established facts, current goal) after this step
    is_terminal: bool = False  # does this step complete the proof/disproof?
    verdict: Optional[str] = None  # "PROVED" | "DISPROVED" if terminal


@runtime_checkable
class Policy(Protocol):
    """The model being trained (served for generation)."""

    def propose_steps(self, problem: Problem, history: list[str], k: int) -> list[str]:
        """Sample k INDEPENDENT candidate next steps from the current state.
        (Independent = siblings do not see each other; keeps MC value estimates clean.)"""
        ...

    def revise_step(
        self, problem: Problem, history: list[str], failed_step: str, reason: str
    ) -> str:
        """In-context retry: produce a revised step given the failed step + why it failed."""
        ...


@runtime_checkable
class Judge(Protocol):
    """Validates a step and maintains the proof state. Swappable (served model)."""

    def judge_step(
        self, problem: Problem, history: list[str], step: str
    ) -> StepJudgement: ...

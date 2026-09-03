"""Deterministic stub Policy/Judge for sandbox testing the engine without a GPU/model.

Scenario (problem.label=True): root branches to move "A" (twice -> merges) and move "B".
  A -> "VERDICT PROVED"  (correct, reward 1)
  B -> "BAD step" (invalid) -> retry -> "VERDICT DISPROVED" (incorrect, reward 0)
This exercises: sibling merge, terminal verdict + endpoint reward, an invalid step,
in-context retry/recovery, and a divergent (critical) root fork.
"""

from __future__ import annotations

from .types import Problem, StepJudgement


class StubPolicy:
    def __init__(self, root_moves=("A", "A", "B")):
        self.root_moves = list(root_moves)
        self._i = 0

    def propose_steps(self, problem: Problem, history: list[str], k: int) -> list[str]:
        if not history:
            move = self.root_moves[self._i % len(self.root_moves)]
            self._i += 1
            return [move]
        if history[-1] == "A":
            return ["VERDICT PROVED"]
        if history[-1] == "B":
            return ["BAD step"]
        return ["VERDICT PROVED"]

    def revise_step(self, problem: Problem, history: list[str], failed_step: str, reason: str) -> str:
        return "VERDICT DISPROVED" if history and history[-1] == "B" else "VERDICT PROVED"

    # batch methods so tests exercise the engine's batched lock-step path
    def propose_steps_batch(self, problem: Problem, histories: list) -> list[str]:
        return [self.propose_steps(problem, h, k=1)[0] for h in histories]

    def revise_steps_batch(self, problem: Problem, items: list) -> list[str]:
        return [self.revise_step(problem, h, fs, r) for (h, fs, r) in items]


class StubJudge:
    def judge_step(self, problem: Problem, history: list[str], step: str) -> StepJudgement:
        if "BAD" in step:
            return StepJudgement(valid=False, reason="invalid: contains BAD")
        summary = "GOAL | " + " > ".join(history + [step])
        if step.upper().startswith("VERDICT"):
            verdict = step.split()[1].upper()
            return StepJudgement(valid=True, reason="ok", state_summary=summary,
                                 is_terminal=True, verdict=verdict)
        return StepJudgement(valid=True, reason="ok", state_summary=summary)

    def judge_steps_batch(self, problem: Problem, items: list) -> list[StepJudgement]:
        return [self.judge_step(problem, h, s) for (h, s) in items]

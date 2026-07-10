"""DeepTheorem proof-quality benchmark using the shared step judge."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from evaluation.benchmarks.base_benchmark import BaseBenchmark, DataSetItem
from evaluation.benchmarks.deeptheorem_eval import _coerce_label
from src.judge.step_judge import DeterministicJudge
from src.training.rl.types import Problem, StepJudgement


_DEFAULT_PATH = "data/deeptheorem_judge_mvp.jsonl"
_QUOTED_RE = re.compile(r'"([^"\r\n]*(?:\\.[^"\r\n]*)*)"', re.DOTALL)
_STEP_RE = re.compile(r"<step>\s*(.*?)\s*</step>", re.IGNORECASE | re.DOTALL)
_VERDICT_RE = re.compile(r"verdict\s*:?\s*(proved|disproved)", re.IGNORECASE)
_STEP_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s+|\d+[.)]\s+|step\s*\d+\s*[:.)-]\s*)", re.IGNORECASE)


def _decode_quoted(raw: str) -> str:
    try:
        return str(json.loads(f'"{raw}"')).strip()
    except json.JSONDecodeError:
        return raw.replace('\\"', '"').strip()


def _line_steps(text: str) -> list[str]:
    steps = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _VERDICT_RE.search(line):
            continue
        line = _STEP_PREFIX_RE.sub("", line).strip()
        if line:
            steps.append(line)
    return steps


class DeepTheoremJudgeEval(BaseBenchmark[str, dict[str, Any]]):
    def __init__(self, verifier=None):
        self.verifier = verifier or DeterministicJudge()

    def load_dataset(self) -> list[DataSetItem[str, bool]]:
        path = os.environ.get("DEEPTHEOREM_JUDGE_EVAL_PATH", _DEFAULT_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DeepTheorem judge eval file not found at '{path}'. "
                "Set DEEPTHEOREM_JUDGE_EVAL_PATH or create the MVP fixture."
            )
        items: list[DataSetItem[str, bool]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                statement = row["statement"] if "statement" in row else row["input"]
                label = _coerce_label(row["label"] if "label" in row else row["output"])
                items.append(
                    DataSetItem(
                        input=statement,
                        output=label,
                        id=str(row.get("id", len(items))),
                        metadata={
                            "domain": row.get("domain", ""),
                            "difficulty": row.get("difficulty"),
                        },
                    )
                )
        return items

    def get_user_prompt(self, input: str) -> str:
        return (
            f"Prove or disprove the following:\n{input}\n\n"
            'Write each proof step either as its own quoted string or inside <step>...</step>. '
            'End with a final line: "Verdict: PROVED" or "Verdict: DISPROVED".'
        )

    def parse_output(self, output: str) -> dict[str, Any]:
        text = output or ""
        quoted = [_decode_quoted(m.group(1)) for m in _QUOTED_RE.finditer(text)]
        q_verdicts = [
            m.group(1).upper()
            for part in quoted
            for m in [_VERDICT_RE.search(part)]
            if m
        ]
        found = [m.upper() for m in _VERDICT_RE.findall(text)]
        verdicts = q_verdicts or found
        verdict = verdicts[-1].upper() if verdicts else None
        steps = [part for part in quoted if part and not _VERDICT_RE.search(part)]
        if not steps:
            steps = [m.strip() for m in _STEP_RE.findall(text) if m.strip() and not _VERDICT_RE.search(m)]
        if not steps and verdicts:
            steps = _line_steps(text)
        return {
            "raw_output": output,
            "steps": steps,
            "verdict": verdict,
            "format_valid": bool(verdicts) and bool(steps),
            "judgement": None,
        }

    def score(self, item: DataSetItem[str, bool], at_output: dict[str, Any]) -> float:
        prob = Problem(id=str(item.id), statement=item.input, label=bool(item.output))
        hist: list[str] = []
        judgements: list[StepJudgement] = []

        for step in at_output["steps"]:
            j = self.verifier.judge_step(prob, hist, step)
            judgements.append(j)
            if j.valid:
                hist.append(step)

        expected = "PROVED" if item.output else "DISPROVED"
        verdict_ok = 1.0 if at_output["verdict"] == expected else 0.0
        format_ok = 1.0 if at_output["format_valid"] else 0.0
        if judgements:
            valid_steps = sum(1 for j in judgements if j.valid)
            proof_ok = valid_steps / len(judgements)
        else:
            valid_steps = 0
            proof_ok = 0.0
        invalid_steps = len(judgements) - valid_steps
        supported = 1.0 if proof_ok == 1.0 and verdict_ok == 1.0 else 0.0

        score = (
            0.50 * verdict_ok
            + 0.30 * proof_ok
            + 0.10 * format_ok
            + 0.10 * supported
        )

        at_output["judgement"] = {
            "score": score,
            "expected_verdict": expected,
            "verdict_correct": verdict_ok,
            "proof_validity": proof_ok,
            "format_validity": format_ok,
            "conclusion_supported": supported,
            "valid_steps": valid_steps,
            "invalid_steps": invalid_steps,
            "step_reasons": [j.reason for j in judgements],
        }
        return score

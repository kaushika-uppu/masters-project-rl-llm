"""Provide the RL pipeline with a list[Problem].

Question selection / subsetting / train-eval disjointness is owned EXTERNALLY. The
pipeline just consumes whatever examples it is given via `load_problems_jsonl(path)`.
"""
from __future__ import annotations

import json
from typing import Optional

from .types import Problem


def _coerce_label(v) -> bool:
    """Robustly coerce a jsonl label to bool. The source truth_value is a real bool, but
    hand-authored files may use strings ("true"/"false") where bool("false") would wrongly
    be True — so normalise explicitly."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "proved"):
            return True
        if s in ("false", "0", "no", "disproved"):
            return False
    raise ValueError(f"Unparseable problem label: {v!r}")


def load_problems_jsonl(path: str, limit: Optional[int] = None) -> list[Problem]:
    """Read a curated jsonl. Each line: {"id","statement","label"[,"reference","domain","difficulty"]}.
    Accepts "input"/"output" as aliases for "statement"/"label".
    Also accepts "tested_variant" and "truth_value" for DeepTheorem format."""
    problems: list[Problem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            # Try multiple possible field names for statement
            if "statement" in d:
                statement = d["statement"]
            elif "input" in d:
                statement = d["input"]
            elif "tested_variant" in d:
                statement = d["tested_variant"]
            elif "informal_theorem" in d:
                statement = d["informal_theorem"]
            else:
                raise KeyError(f"Problem entry missing statement field. Has keys: {list(d.keys())}")

            # Try multiple possible field names for label
            if "label" in d:
                label = d["label"]
            elif "output" in d:
                label = d["output"]
            elif "truth_value" in d:
                label = d["truth_value"]
            else:
                raise KeyError(f"Problem entry missing label field. Has keys: {list(d.keys())}")

            problems.append(Problem(
                id=str(d.get("id", len(problems))),
                statement=statement,
                label=_coerce_label(label),
                reference=d.get("reference", ""),
                domain=d.get("domain", ""),
                difficulty=d.get("difficulty"),
            ))
            if limit and len(problems) >= limit:
                break
    return problems

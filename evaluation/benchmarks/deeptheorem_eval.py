"""In-domain DeepTheorem eval: prove-or-disprove verdict accuracy.

Runs on a PROVIDED jsonl of examples (question selection / train-eval disjointness is
external). Point it at the file via env var DEEPTHEOREM_EVAL_PATH (default
data/deeptheorem_eval.jsonl). Each line: {"id","statement","label"} where label is a
bool (True=PROVED). "input"/"output" accepted as aliases.
"""
from __future__ import annotations

import json
import os
import re
from typing import List, Optional

from evaluation.benchmarks.base_benchmark import BaseBenchmark, DataSetItem

_DEFAULT_PATH = "data/deeptheorem_eval.jsonl"


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
    raise ValueError(f"Unparseable DeepTheorem label: {v!r}")


class DeepTheoremEval(BaseBenchmark[str, bool]):
    def load_dataset(self) -> List[DataSetItem[str, bool]]:
        path = os.environ.get("DEEPTHEOREM_EVAL_PATH", _DEFAULT_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"DeepTheorem eval file not found at '{path}'. Provide a curated jsonl "
                f"(fields: id, statement, label) or set DEEPTHEOREM_EVAL_PATH."
            )
        items: List[DataSetItem[str, bool]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                statement = d["statement"] if "statement" in d else d["input"]
                label = _coerce_label(d["label"] if "label" in d else d["output"])

                metadata = {
                    "domain": d.get("domain", ""), 
                    "difficulty": d.get("difficulty"),
                    "base_id": d.get("base_id", ""),
                    "variant_type": d.get("variant_type", "")
                }
                items.append(DataSetItem(
                    input=statement, output=label, id=str(d.get("id", len(items))),
                    metadata=metadata,
                ))
        return items

    def get_user_prompt(self, input: str) -> str:
        return (
            f"Prove or disprove the following:\n{input}\n\n"
            "Reason step by step, wrapping each step in <step>...</step>. "
            "End with a final line exactly: 'Verdict: PROVED' or 'Verdict: DISPROVED'."
        )

    def parse_output(self, output: str) -> Optional[bool]:
        m = re.findall(r"verdict\s*:?\s*(proved|disproved)", output, re.IGNORECASE)
        if not m:
            return None
        return m[-1].lower() == "proved"

    def score(self, item: DataSetItem[str, bool], at_output: Optional[bool]) -> float:
        if item.output is None or at_output is None:
            return 0.0
        return 1.0 if bool(item.output) == bool(at_output) else 0.0
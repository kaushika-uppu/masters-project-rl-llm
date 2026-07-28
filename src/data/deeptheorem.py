"""DeepTheorem data layer.

Loads the DeepTheorem dataset and exposes:
  - difficulty split (<=7.0 SFT / >7.0 RL pool),
  - matched true/false "prove or disprove" variant pairs (verifiable endpoint labels),
  - <step>-tagged formatting consistent across SFT and RL.

Schema (confirmed, Jiahao004/DeepTheorem): id, source, ori_question, ori_solution,
domain (list[str]), difficulty (float ~5-10), rationale, informal_theorem,
informal_theorem_qa, proof, truth_value (bool), pos, neg. `pos`/`neg` are dicts
{question: str, response: str, truth_value: bool} holding the matched true/false
prove-or-disprove variants — labels are explicit (per-variant truth_value), no inference
needed. Name-based fallbacks below are kept only for robustness; the primary path uses
these confirmed columns. (`scripts/inspect_deeptheorem.py` re-verifies on the cluster.)
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# Instruct a verdict (prove/disprove) AND step-by-step justification. The verdict is the
# verifiable endpoint signal; the steps are what we actually train on.
PROVE_OR_DISPROVE_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. Decide whether the statement is true "
    "or false, then justify it with a rigorous step-by-step argument. Wrap each "
    "reasoning step in <step>...</step> tags. End with a final line: "
    "'Verdict: PROVED' if the statement is true, or 'Verdict: DISPROVED' if it is false."
)

# Fallback column-name candidates (first match wins). Confirm on cluster.
_STATEMENT_KEYS = ("informal_theorem", "statement", "theorem", "informal_statement", "question")
_PROOF_KEYS = ("proof", "informal_proof", "solution", "response")
_DIFFICULTY_KEYS = ("difficulty", "level", "hardness")
_VARIANT_QUESTION_KEYS = ("question", "statement", "theorem", "prompt")
_VARIANT_RESPONSE_KEYS = ("response", "proof", "solution", "answer")
_VARIANT_LABEL_KEYS = ("label", "is_true", "truth", "veracity", "answer_bool")


@dataclass
class DeepTheoremColumns:
    """Resolved column names for the DeepTheorem HF schema.

    Confirmed schema (Jiahao004/DeepTheorem): id, source, ori_question, ori_solution,
    domain, difficulty, rationale, informal_theorem, informal_theorem_qa, proof,
    truth_value, pos, neg. The true/false variants live in the separate `pos` (true)
    and `neg` (false) columns — labels are explicit by column, no inference needed.
    """
    statement: str = "informal_theorem"
    proof: str = "proof"
    difficulty: str = "difficulty"
    truth_value: str = "truth_value"
    pos: str = "pos"           # positive (true) prove-or-disprove variant
    neg: str = "neg"           # negated (false) prove-or-disprove variant
    domain: str = "domain"
    variants: Optional[str] = None  # legacy: single list-of-dicts column (fallback)


@dataclass
class Variant:
    statement: str            # the "prove or disprove" claim
    label: bool               # True = statement is true (PROVED), False = DISPROVED
    reference: str            # worked reference response (used offline only, not as live reward)
    is_original: bool = False # True for the dataset's always-true original theorem
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------------------
# Loading & difficulty split
# --------------------------------------------------------------------------------------
def load_deeptheorem(split: str = "train", name: str = "Jiahao004/DeepTheorem"):
    """Load the HF dataset (cluster/network required)."""
    from datasets import load_dataset
    return load_dataset(name, split=split)


def _first_present(row: dict, keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] is not None:
            return k
    return None


def resolve_difficulty_key(row: dict, cols: DeepTheoremColumns) -> Optional[str]:
    return cols.difficulty or _first_present(row, _DIFFICULTY_KEYS)


def split_by_difficulty(ds, threshold: float = 7.0, cols: Optional[DeepTheoremColumns] = None):
    """Return (easy, hard) datasets split at `threshold` (scale ~4-10; SFT<=7, RL>7)."""
    cols = cols or DeepTheoremColumns()
    key = cols.difficulty or _first_present(ds[0], _DIFFICULTY_KEYS)
    if key is None:
        raise KeyError("Could not find a difficulty column; set DeepTheoremColumns.difficulty")
    easy = ds.filter(lambda r: r[key] is not None and float(r[key]) <= threshold)
    hard = ds.filter(lambda r: r[key] is not None and float(r[key]) > threshold)
    return easy, hard


# --------------------------------------------------------------------------------------
# Variant parsing (the verifiable true/false pairs)
# --------------------------------------------------------------------------------------
def _coerce_variant_item(item: Any) -> Optional[dict]:
    """A variant item may be a dict or a JSON string. Return a dict or None."""
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        try:
            obj = json.loads(item)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _looks_like_variant_list(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or not value:
        return False
    d = _coerce_variant_item(value[0])
    if not d:
        return False
    has_q = any(k in d for k in _VARIANT_QUESTION_KEYS)
    has_r = any(k in d for k in _VARIANT_RESPONSE_KEYS)
    return has_q and has_r


def find_variant_column(row: dict, cols: Optional[DeepTheoremColumns] = None) -> Optional[str]:
    cols = cols or DeepTheoremColumns()
    if cols.variants and cols.variants in row:
        return cols.variants
    for k, v in row.items():
        if _looks_like_variant_list(v):
            return k
    return None


def _infer_label_from_response(response: str) -> Optional[bool]:
    """Fallback: read the verdict from the reference response text."""
    if not response:
        return None
    tail = response[-400:].lower()
    disproved = any(w in tail for w in ("disprov", "is false", "does not hold", "counterexample", "contradiction, so the statement is false"))
    proved = any(w in tail for w in ("proved", "is true", "holds", "qed", "∎", "thus the statement holds"))
    if disproved and not proved:
        return False
    if proved and not disproved:
        return True
    return None


def _variant_from_value(value: Any, label: bool) -> Optional[Variant]:
    """Build a Variant from a `pos`/`neg` cell (dict, JSON string, or plain string).

    The confirmed schema gives each variant dict its own `truth_value` bool, so we read
    that as the label when present and only fall back to the positional `label`
    (pos=True, neg=False) when it is missing.
    """
    d = _coerce_variant_item(value)
    if d:
        qk = _first_present(d, _VARIANT_QUESTION_KEYS)
        rk = _first_present(d, _VARIANT_RESPONSE_KEYS)
        statement = str(d[qk]) if qk else None
        reference = str(d[rk]) if rk else ""
        tv = _as_bool(d["truth_value"]) if "truth_value" in d else None
        resolved = tv if tv is not None else label
        return Variant(statement=statement, label=resolved, reference=reference) if statement else None
    if isinstance(value, str) and value.strip():
        return Variant(statement=value.strip(), label=label, reference="")
    return None


def parse_variants(
    row: dict,
    cols: Optional[DeepTheoremColumns] = None,
    label_strategy: str = "auto",   # only used for the legacy list-column fallback
    include_original: bool = False,
    warn: bool = True,
) -> list[Variant]:
    """Extract matched true/false prove-or-disprove variants from a row.

    Primary path: explicit `pos` (true) / `neg` (false) columns — labels are by column.
    Fallback: a legacy single list-of-dicts column (uses label_strategy).
    include_original: also emit the always-(usually)-true original (informal_theorem +
        proof). Off by default to keep the true/false verdict balance from pos/neg.
    """
    cols = cols or DeepTheoremColumns()
    out: list[Variant] = []

    pos_key = cols.pos if cols.pos in row else None
    neg_key = cols.neg if cols.neg in row else None

    if pos_key or neg_key:
        if pos_key and row.get(pos_key) is not None:
            v = _variant_from_value(row[pos_key], True)
            if v:
                v.meta["variant"] = "pos"; out.append(v)
        if neg_key and row.get(neg_key) is not None:
            v = _variant_from_value(row[neg_key], False)
            if v:
                v.meta["variant"] = "neg"; out.append(v)
    else:
        # legacy fallback: single list-of-dicts column
        vcol = find_variant_column(row, cols)
        if vcol is not None:
            items = [d for d in (_coerce_variant_item(x) for x in row[vcol]) if d]
            for idx, d in enumerate(items):
                qk = _first_present(d, _VARIANT_QUESTION_KEYS)
                rk = _first_present(d, _VARIANT_RESPONSE_KEYS)
                if qk is None or rk is None:
                    continue
                label = _resolve_label(d, idx, d.get(rk, ""), label_strategy, warn)
                if label is None:
                    continue
                out.append(Variant(statement=str(d[qk]), label=label, reference=str(d[rk]),
                                   meta={"variant_index": idx}))

    if include_original:
        sk = cols.statement if cols.statement in row else _first_present(row, _STATEMENT_KEYS)
        pk = cols.proof if cols.proof in row else _first_present(row, _PROOF_KEYS)
        if sk and pk and row.get(sk) and row.get(pk):
            tv = row.get(cols.truth_value)
            orig_label = _as_bool(tv) if tv is not None else True
            if orig_label is None:
                orig_label = True
            out.append(Variant(statement=str(row[sk]), label=orig_label, reference=str(row[pk]),
                               is_original=True, meta={}))
    return out


def _resolve_label(d: dict, idx: int, response: str, strategy: str, warn: bool) -> Optional[bool]:
    lk = _first_present(d, _VARIANT_LABEL_KEYS)
    if strategy in ("field", "auto") and lk is not None:
        return _as_bool(d[lk])
    if strategy in ("infer", "auto"):
        inferred = _infer_label_from_response(response)
        if inferred is not None:
            return inferred
    if strategy in ("positional", "auto"):
        if warn:
            warnings.warn(
                "DeepTheorem variant labels resolved POSITIONALLY (0->True,1->False). "
                "Confirm the real label encoding via scripts/inspect_deeptheorem.py.",
                stacklevel=2,
            )
        return idx == 0
    return None


def _as_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes", "proved", "provable"):
            return True
        if s in ("false", "0", "no", "disproved"):
            return False
    return None


# --------------------------------------------------------------------------------------
# Formatting (shared by SFT and RL so there is no train->RL distribution shift)
# --------------------------------------------------------------------------------------
_PARA = re.compile(r"\n\s*\n")


def to_step_format(text: str) -> str:
    """Split a proof on blank lines and wrap each paragraph as a <step>."""
    paras = [p.strip() for p in _PARA.split(text.strip()) if p.strip()]
    return "\n".join(f"<step>{p}</step>" for p in paras)


def build_sft_examples(
    rows: Iterable[dict],
    cols: Optional[DeepTheoremColumns] = None,
    include_original: bool = False,
    label_strategy: str = "auto",
    append_verdict: bool = True,
) -> list[dict]:
    """Expand rows into prove-or-disprove SFT chat examples (one per variant)."""
    cols = cols or DeepTheoremColumns()
    examples: list[dict] = []
    for row in rows:
        for v in parse_variants(row, cols, label_strategy, include_original, warn=False):
            steps = to_step_format(v.reference)
            if append_verdict:
                steps = f"{steps}\nVerdict: {'PROVED' if v.label else 'DISPROVED'}"
            examples.append({
                "messages": [
                    {"role": "system", "content": PROVE_OR_DISPROVE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Prove or disprove the following:\n{v.statement}"},
                    {"role": "assistant", "content": steps},
                ],
                "label": v.label,
                "is_original": v.is_original,
            })
    return examples

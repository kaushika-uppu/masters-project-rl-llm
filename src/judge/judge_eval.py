"""Reliability evaluation for the RL step judge (torch-free).

Builds a labeled set of (statement, history, step) items from reference proofs
(gold steps -> should be accepted; corrupted steps -> should be rejected) and
computes classification metrics from the judge's StepJudgement outputs.

CAVEAT: the synthetic corruptions are heuristic. Some of them (notably "negated"
and "numeric_perturb") can accidentally still yield a valid step (e.g. flipping
"is" in a sentence that is not load-bearing, or perturbing a number in a
side remark), and gold steps can be judged invalid out of context. That is why we
report PER-KIND accept rates instead of treating any single kind as perfect
ground truth; the clear-cut kinds (premature_verdict, wrong_verdict, degenerate)
are the most trustworthy signals.
"""
from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from src.data.deeptheorem import to_step_format

PARSE_ERROR_REASON = "judge parse error"

GOLD_KINDS = ("gold", "gold_terminal")
CORRUPTION_KINDS = (
    "wrong_verdict",
    "premature_verdict",
    "skip_to_conclusion",
    "negated",
    "numeric_perturb",
    "off_topic",
    "degenerate",
    "circular",
)

DEGENERATE_STRINGS = (
    "/\n\n\n,\n,\n,\n,",
    "\\FAMILY\\FAMILY\\FAMILY",
    "",
    "<step></step>",
)


@dataclass
class EvalItem:
    item_id: str
    problem_id: str
    statement: str
    label: Optional[bool]
    history: list
    step: str
    gold_valid: bool
    gold_terminal: Optional[bool] = None
    gold_verdict: Optional[str] = None
    kind: str = "labeled"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- steps

_STEP_RE = re.compile(r"<step>(.*?)</step>", re.S)


def reference_steps(reference: str) -> list:
    """Split a reference proof into step texts (paragraph split via to_step_format)."""
    if not reference or not reference.strip():
        return []
    tagged = reference if "<step>" in reference else to_step_format(reference)
    return [s.strip() for s in _STEP_RE.findall(tagged) if s.strip()]


def verdict_line(label: bool) -> str:
    return "Verdict: PROVED" if label else "Verdict: DISPROVED"


def verdict_word(label: bool) -> str:
    return "PROVED" if label else "DISPROVED"


# ------------------------------------------------------------------ corruption rules

def _swap_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _word_swap(a: str, b: str):
    pat = re.compile(rf"\b({a}|{b})\b", re.I)

    def fn(m: "re.Match") -> str:
        w = m.group(1)
        return _swap_case(w, b if w.lower() == a else a)

    return pat, fn


# Ordered: the first rule whose pattern matches is applied (first occurrence only).
_NEGATE_RULES = [
    (re.compile(r"≤"), ">"),
    (re.compile(r"≥"), "<"),
    (re.compile(r"(?<![-=<])<(?![a-zA-Z/=])"), "≥"),
    (re.compile(r"(?<![-=>])>(?!=)"), "≤"),
    (re.compile(r"(?<![<>≤≥!=:])=(?!=)"), "≠"),
    (re.compile(r" is (?!not )"), " is not "),
    (re.compile(r" not "), " "),
    _word_swap("all", "some"),
    _word_swap("true", "false"),
    _word_swap("increasing", "decreasing"),
    _word_swap("even", "odd"),
]


def negate_step(step: str) -> Optional[str]:
    """Deterministic logical/relational flip of the first matching rule; None if none match."""
    for pat, repl in _NEGATE_RULES:
        m = pat.search(step)
        if not m:
            continue
        if callable(repl):
            out = pat.sub(repl, step, count=1)
        else:
            out = step[: m.start()] + repl + step[m.end():]
        if out != step:
            return out
    return None


_NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def perturb_number(step: str, rng: random.Random) -> Optional[str]:
    """Change one number in the step (+1 or x2); None if it has no number."""
    ms = list(_NUM_RE.finditer(step))
    if not ms:
        return None
    m = rng.choice(ms)
    txt = m.group(0)
    if "." in txt:
        val = float(txt)
        new = f"{val + 1:g}" if rng.random() < 0.5 else f"{val * 2:g}"
    else:
        val = int(txt)
        new = str(val + 1) if rng.random() < 0.5 else str(val * 2)
        if new == txt:  # 0 * 2
            new = str(val + 1)
    out = step[: m.start()] + new + step[m.end():]
    return out if out != step else None


# --------------------------------------------------------------------------- builder

def _rng(seed: int, *parts: Any) -> random.Random:
    return random.Random(":".join([str(seed), *map(str, parts)]))


def build_eval_items_with_stats(
    problems: Iterable,
    n_per_problem: int = 2,
    corruptions: Optional[Iterable[str]] = None,
    seed: int = 42,
):
    """Build the labeled set. Returns (items, stats). `problems` need .id/.statement/.label/.reference."""
    kinds = list(CORRUPTION_KINDS if corruptions is None else corruptions)
    unknown = [k for k in kinds if k not in CORRUPTION_KINDS]
    if unknown:
        raise ValueError(f"Unknown corruption kinds {unknown}; valid: {list(CORRUPTION_KINDS)}")

    problems = list(problems)
    usable = []
    skipped_no_ref = 0
    for p in problems:
        steps = reference_steps(getattr(p, "reference", "") or "")
        if not steps:
            skipped_no_ref += 1
            continue
        usable.append((p, steps))

    items: list = []
    skipped_kind: dict = {k: 0 for k in kinds}
    one_step_refs = sum(1 for _, s in usable if len(s) == 1)

    def add(p, kind, idx, history, step, gold_valid, **kw):
        items.append(EvalItem(
            item_id=f"{p.id}:{kind}:{idx}", problem_id=str(p.id), statement=p.statement,
            label=bool(p.label), history=list(history), step=step, gold_valid=gold_valid,
            kind=kind, **kw))

    for pi, (p, steps) in enumerate(usable):
        n = len(steps)
        label = bool(p.label)
        rng = _rng(seed, p.id, "pos")

        # gold positives
        positions = list(range(n))
        rng.shuffle(positions)
        for j, i in enumerate(sorted(positions[: max(0, n_per_problem)])):
            add(p, "gold", j, steps[:i], steps[i], True)
        # gold terminal positive
        add(p, "gold_terminal", 0, steps, verdict_line(label), True,
            gold_terminal=True, gold_verdict=verdict_word(label))

        for kind in kinds:
            r = _rng(seed, p.id, kind)
            if kind == "wrong_verdict":
                add(p, kind, 0, steps, verdict_line(not label), False)
            elif kind == "premature_verdict":
                add(p, kind, 0, [], verdict_line(label), False)
            elif kind == "skip_to_conclusion":
                if n >= 4:
                    add(p, kind, 0, steps[:1], steps[-1], False)
                else:
                    skipped_kind[kind] += 1
            elif kind == "negated":
                order = list(range(n))
                r.shuffle(order)
                done = False
                for i in order:
                    neg = negate_step(steps[i])
                    if neg is not None:
                        add(p, kind, 0, steps[:i], neg, False)
                        done = True
                        break
                if not done:
                    skipped_kind[kind] += 1
            elif kind == "numeric_perturb":
                order = list(range(n))
                r.shuffle(order)
                done = False
                for i in order:
                    pert = perturb_number(steps[i], r)
                    if pert is not None:
                        add(p, kind, 0, steps[:i], pert, False)
                        done = True
                        break
                if not done:
                    skipped_kind[kind] += 1
            elif kind == "off_topic":
                others = [(q, s) for q, s in usable if q.id != p.id]
                if not others:
                    skipped_kind[kind] += 1
                    continue
                _, osteps = r.choice(others)
                i = r.randrange(n)
                add(p, kind, 0, steps[:i], r.choice(osteps), False)
            elif kind == "degenerate":
                add(p, kind, 0, steps[: r.randrange(n)], r.choice(DEGENERATE_STRINGS), False)
            elif kind == "circular":
                i = r.randrange(n)
                claim = p.statement.strip().rstrip(". ")
                add(p, kind, 0, steps[:i], f"Therefore, {claim} holds.", False)

    stats = {
        "n_problems": len(problems),
        "n_usable": len(usable),
        "n_skipped_no_reference": skipped_no_ref,
        "n_single_step_references": one_step_refs,
        "skipped_by_kind": skipped_kind,
        "n_items": len(items),
    }
    return items, stats


def build_eval_items(problems, n_per_problem: int = 2, corruptions=None, seed: int = 42) -> list:
    return build_eval_items_with_stats(problems, n_per_problem, corruptions, seed)[0]


def load_labeled_items(path: str) -> list:
    """Hand-labeled jsonl: {statement, label?, history: [str], step, gold_valid, kind?}.
    Optional: id/problem_id/item_id, gold_terminal, gold_verdict."""
    items = []
    with open(path) as f:
        for ln, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for req in ("statement", "step", "gold_valid"):
                if req not in d:
                    raise KeyError(f"{path}:{ln + 1} missing field '{req}'")
            label = d.get("label")
            if label is not None:
                from src.training.rl.problems import _coerce_label
                label = _coerce_label(label)
            pid = str(d.get("problem_id", d.get("id", f"labeled{ln}")))
            items.append(EvalItem(
                item_id=str(d.get("item_id", f"{pid}:labeled:{ln}")),
                problem_id=pid,
                statement=d["statement"],
                label=label,
                history=list(d.get("history") or []),
                step=d["step"],
                gold_valid=bool(d["gold_valid"]),
                gold_terminal=d.get("gold_terminal"),
                gold_verdict=d.get("gold_verdict"),
                kind=str(d.get("kind", "labeled")),
            ))
    return items


# --------------------------------------------------------------------------- metrics

def wilson_ci(k: int, n: int, z: float = 1.96) -> Optional[list]:
    """Wilson score interval for a binomial proportion; None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _rate(k: int, n: int) -> Optional[float]:
    return k / n if n else None


def compute_metrics(items: list, judgements: list) -> dict:
    if len(items) != len(judgements):
        raise ValueError("items and judgements must have equal length")
    n = len(items)
    parse_err = sum(1 for j in judgements if (j.reason or "") == PARSE_ERROR_REASON)
    correct = sum(1 for it, j in zip(items, judgements) if bool(j.valid) == it.gold_valid)

    tp = fp = fn = tn = 0  # positive class = "invalid"
    for it, j in zip(items, judgements):
        pred_inv, gold_inv = not j.valid, not it.gold_valid
        if pred_inv and gold_inv:
            tp += 1
        elif pred_inv and not gold_inv:
            fp += 1
        elif not pred_inv and gold_inv:
            fn += 1
        else:
            tn += 1
    precision = _rate(tp, tp + fp)
    recall = _rate(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    n_gold = tn + fp
    per_kind: dict = {}
    for kind in sorted({it.kind for it in items}):
        pairs = [(it, j) for it, j in zip(items, judgements) if it.kind == kind]
        acc = sum(1 for _, j in pairs if j.valid)
        per_kind[kind] = {
            "n": len(pairs),
            "n_accepted": acc,
            "accept_rate": _rate(acc, len(pairs)),
            "accept_ci95": wilson_ci(acc, len(pairs)),
            "gold_valid": pairs[0][0].gold_valid if pairs else None,
        }

    term = [(it, j) for it, j in zip(items, judgements) if it.gold_terminal]
    term_n = len(term)
    term_is = sum(1 for _, j in term if j.is_terminal)
    term_ok = sum(1 for it, j in term
                  if j.verdict is not None and it.gold_verdict is not None
                  and str(j.verdict).upper() == str(it.gold_verdict).upper())

    return {
        "n": n,
        "parse_error_rate": _rate(parse_err, n),
        "n_parse_errors": parse_err,
        "accuracy": _rate(correct, n),
        "accuracy_ci95": wilson_ci(correct, n),
        "precision_invalid": precision,
        "recall_invalid": recall,
        "f1_invalid": f1,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "false_reject_rate": _rate(fp, n_gold),
        "false_reject_ci95": wilson_ci(fp, n_gold),
        "n_gold": n_gold,
        "per_kind": per_kind,
        "terminal": {
            "n": term_n,
            "is_terminal_rate": _rate(term_is, term_n),
            "verdict_accuracy": _rate(term_ok, term_n),
            "verdict_accuracy_ci95": wilson_ci(term_ok, term_n),
        },
    }


def _pct(x) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def _ci(ci) -> str:
    return "" if not ci else f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]"


def format_report(metrics: dict, title: str = "Judge reliability report") -> str:
    m = metrics
    L = [title, "=" * len(title)]
    L.append(f"items: {m['n']}   parse errors: {m['n_parse_errors']} ({_pct(m['parse_error_rate']).strip()})")
    L.append(f"accuracy:            {_pct(m['accuracy'])}  95% CI {_ci(m['accuracy_ci95'])}")
    L.append(f"precision (invalid): {_pct(m['precision_invalid'])}")
    L.append(f"recall (invalid):    {_pct(m['recall_invalid'])}   <- bad-step catch rate")
    L.append(f"f1 (invalid):        {_pct(m['f1_invalid'])}")
    L.append(f"false-reject (gold): {_pct(m['false_reject_rate'])}  95% CI {_ci(m['false_reject_ci95'])}  (n_gold={m['n_gold']})")
    c = m["confusion"]
    L.append(f"confusion (pos=invalid): TP={c['tp']} FP={c['fp']} FN={c['fn']} TN={c['tn']}")
    L.append("")
    L.append(f"{'kind':<22}{'gold':>6}{'n':>6}{'accepted':>10}{'accept%':>9}  95% CI")
    for kind, d in m["per_kind"].items():
        gold = "valid" if d["gold_valid"] else "bad"
        L.append(f"{kind:<22}{gold:>6}{d['n']:>6}{d['n_accepted']:>10}{_pct(d['accept_rate']):>9}  {_ci(d['accept_ci95'])}")
    L.append("(accept% = fraction judged valid; want HIGH for gold, LOW for bad kinds)")
    t = m["terminal"]
    L.append("")
    L.append(f"terminal items: {t['n']}  is_terminal rate: {_pct(t['is_terminal_rate']).strip()}  "
             f"verdict accuracy: {_pct(t['verdict_accuracy']).strip()} {_ci(t['verdict_accuracy_ci95'])}")
    return "\n".join(L)


def _kind_accept(m: dict, kind: str) -> Optional[float]:
    d = m["per_kind"].get(kind)
    return d["accept_rate"] if d else None


def format_comparison(rows: dict) -> str:
    """rows: {judge_name: metrics}. One row per judge."""
    cols = [
        ("accuracy", lambda m: m["accuracy"]),
        ("recall_bad", lambda m: m["recall_invalid"]),
        ("false_reject", lambda m: m["false_reject_rate"]),
        ("premature_acc", lambda m: _kind_accept(m, "premature_verdict")),
        ("wrong_vrd_acc", lambda m: _kind_accept(m, "wrong_verdict")),
        ("degenerate_acc", lambda m: _kind_accept(m, "degenerate")),
        ("parse_err", lambda m: m["parse_error_rate"]),
    ]
    w = max([len("judge")] + [len(k) for k in rows]) + 2
    head = f"{'judge':<{w}}" + "".join(f"{c:>15}" for c, _ in cols)
    L = [head, "-" * len(head)]
    for name, m in rows.items():
        L.append(f"{name:<{w}}" + "".join(f"{_pct(f(m)):>15}" for _, f in cols))
    return "\n".join(L)


# --------------------------------------------------------------------------- running

def run_judge(judge, items: list, batch_size: int = 16):
    """Run `judge` over items (grouped by statement, chunked). Returns (judgements, raws).
    raws are '' when the judge cannot expose raw output (e.g. DeterministicJudge)."""
    import inspect
    from src.training.rl.types import Problem

    try:
        has_raw = "return_raw" in inspect.signature(judge.judge_steps_batch).parameters
    except (TypeError, ValueError):
        has_raw = False

    judgements: list = [None] * len(items)
    raws: list = [""] * len(items)
    groups: dict = {}
    for idx, it in enumerate(items):
        groups.setdefault(it.statement, []).append(idx)
    bs = max(1, int(batch_size))
    for statement, idxs in groups.items():
        problem = Problem(id=items[idxs[0]].problem_id, statement=statement,
                          label=bool(items[idxs[0]].label))
        for s in range(0, len(idxs), bs):
            chunk = idxs[s:s + bs]
            pairs = [(items[i].history, items[i].step) for i in chunk]
            if has_raw:
                js, rs = judge.judge_steps_batch(problem, pairs, return_raw=True)
            else:
                js, rs = judge.judge_steps_batch(problem, pairs), [""] * len(chunk)
            for i, j, r in zip(chunk, js, rs):
                judgements[i], raws[i] = j, r
    return judgements, raws

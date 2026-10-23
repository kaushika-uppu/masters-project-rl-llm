import json
import random
from types import SimpleNamespace as NS

import pytest

from scripts import eval_judge
from src.judge import judge_eval as je
from src.training.rl.types import StepJudgement

REF = "Let x be even.\n\nThen x = 2k for some integer k.\n\nSo x + 1 = 2k + 1 is odd.\n\nHence the claim is true."


def prob(pid, ref=REF, label=True, statement="Prove that x is even implies x+1 odd."):
    return NS(id=pid, statement=statement, label=label, reference=ref)


def by_kind(items, kind):
    return [i for i in items if i.kind == kind]


def test_reference_steps_split():
    assert len(je.reference_steps(REF)) == 4
    assert je.reference_steps("") == []


def test_gold_and_terminal_positives():
    items = je.build_eval_items([prob("a")], n_per_problem=2, corruptions=[])
    gold = by_kind(items, "gold")
    steps = je.reference_steps(REF)
    assert len(gold) == 2
    for g in gold:
        assert g.gold_valid and g.step in steps
        assert g.history == steps[: steps.index(g.step)]
    t = by_kind(items, "gold_terminal")[0]
    assert t.history == steps and t.step == "Verdict: PROVED"
    assert t.gold_terminal and t.gold_verdict == "PROVED"
    t2 = by_kind(je.build_eval_items([prob("b", label=False)], corruptions=[]), "gold_terminal")[0]
    assert t2.step == "Verdict: DISPROVED"


def test_negative_kinds():
    ps = [prob("a"), prob("b", ref="Alpha beta.\n\nGamma 3 delta.\n\nEpsilon.\n\nZeta.\n\nEta.")]
    items = je.build_eval_items(ps, corruptions=None, seed=1)
    for it in items:
        if it.kind in je.CORRUPTION_KINDS:
            assert it.gold_valid is False
    pre = by_kind(items, "premature_verdict")[0]
    assert pre.history == [] and pre.step == "Verdict: PROVED"
    wv = by_kind(items, "wrong_verdict")[0]
    assert wv.step == "Verdict: DISPROVED" and wv.history == je.reference_steps(REF)
    skip = [i for i in by_kind(items, "skip_to_conclusion") if i.problem_id == "a"][0]
    assert skip.history == ["Let x be even."] and skip.step == "Hence the claim is true."
    ot = [i for i in by_kind(items, "off_topic") if i.problem_id == "a"][0]
    other = je.reference_steps(ps[1].reference)
    assert ot.step in other
    assert by_kind(items, "degenerate")[0].step in je.DEGENERATE_STRINGS
    circ = by_kind(items, "circular")[0]
    assert circ.step.startswith("Therefore, ") and circ.step.endswith(" holds.")
    num = by_kind(items, "numeric_perturb")
    assert num


def test_negated_changes_and_skips():
    assert je.negate_step("a ≤ b") == "a > b"
    assert je.negate_step("x = 2") == "x ≠ 2"
    assert je.negate_step("x is even") == "x is not even"
    assert je.negate_step("x is not big") == "x is big"
    assert je.negate_step("It is true") != "It is true"
    assert je.negate_step("Hello world.") is None
    items = je.build_eval_items([prob("n", ref="Hello world.\n\nFoo bar.")], corruptions=["negated"])
    assert not by_kind(items, "negated")
    _, stats = je.build_eval_items_with_stats([prob("n", ref="Hello world.")], corruptions=["negated"])
    assert stats["skipped_by_kind"]["negated"] == 1


def test_skip_to_conclusion_needs_four_steps():
    items = je.build_eval_items([prob("s", ref="a1\n\nb2\n\nc3")], corruptions=["skip_to_conclusion"])
    assert not by_kind(items, "skip_to_conclusion")


def test_numeric_perturb_changes():
    out = je.perturb_number("we have 7 apples", random.Random(0))
    assert out != "we have 7 apples" and "7" not in out
    assert je.perturb_number("no digits", random.Random(0)) is None


def test_off_topic_needs_other_problem():
    items = je.build_eval_items([prob("only")], corruptions=["off_topic"])
    assert not by_kind(items, "off_topic")


def test_skips_missing_reference_and_unknown_kind():
    items, stats = je.build_eval_items_with_stats([prob("a"), prob("x", ref="")], corruptions=[])
    assert stats["n_skipped_no_reference"] == 1 and stats["n_usable"] == 1
    assert all(i.problem_id == "a" for i in items)
    with pytest.raises(ValueError):
        je.build_eval_items([prob("a")], corruptions=["bogus"])


def test_determinism():
    ps = [prob("a"), prob("b")]
    a = [i.to_dict() for i in je.build_eval_items(ps, seed=7)]
    b = [i.to_dict() for i in je.build_eval_items(ps, seed=7)]
    assert a == b


def J(valid, reason="ok", term=False, verdict=None):
    return StepJudgement(valid=valid, reason=reason, is_terminal=term, verdict=verdict)


def E(kind, gold_valid, **kw):
    return je.EvalItem(item_id=kind, problem_id="p", statement="s", label=True, history=[], step="x",
                       gold_valid=gold_valid, kind=kind, **kw)


def test_compute_metrics():
    items = [
        E("gold", True), E("gold", True), E("gold", True), E("gold", True),
        E("gold_terminal", True, gold_terminal=True, gold_verdict="PROVED"),
        E("premature_verdict", False), E("premature_verdict", False),
        E("negated", False), E("negated", False),
    ]
    js = [
        J(True), J(True), J(True), J(False),                       # 1 false reject
        J(True, term=True, verdict="PROVED"),
        J(True), J(False),                                         # premature: 1 accepted
        J(False, reason=je.PARSE_ERROR_REASON), J(False),          # negated both caught
    ]
    m = je.compute_metrics(items, js)
    assert m["n"] == 9 and m["n_parse_errors"] == 1
    assert m["parse_error_rate"] == pytest.approx(1 / 9)
    assert m["accuracy"] == pytest.approx(7 / 9)
    c = m["confusion"]
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (3, 1, 1, 4)
    assert m["precision_invalid"] == pytest.approx(3 / 4)
    assert m["recall_invalid"] == pytest.approx(3 / 4)
    assert m["f1_invalid"] == pytest.approx(0.75)
    assert m["false_reject_rate"] == pytest.approx(1 / 5)
    assert m["per_kind"]["premature_verdict"]["accept_rate"] == 0.5
    assert m["per_kind"]["premature_verdict"]["n"] == 2
    assert m["per_kind"]["negated"]["accept_rate"] == 0
    assert m["terminal"]["n"] == 1 and m["terminal"]["is_terminal_rate"] == 1
    assert m["terminal"]["verdict_accuracy"] == 1
    lo, hi = m["accuracy_ci95"]
    assert lo < 7 / 9 < hi
    assert "premature_verdict" in je.format_report(m)
    assert "stub" in je.format_comparison({"stub": m})


def test_wilson():
    assert je.wilson_ci(0, 0) is None
    lo, hi = je.wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)
    assert je.wilson_ci(10, 10)[1] == 1.0


def test_load_labeled(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(json.dumps({"statement": "S", "label": "true", "history": ["a"], "step": "b",
                             "gold_valid": False, "kind": "rollout"}) + "\n\n"
                 + json.dumps({"statement": "S2", "step": "c", "gold_valid": True}) + "\n")
    items = je.load_labeled_items(str(p))
    assert len(items) == 2 and items[0].label is True and items[0].history == ["a"]
    assert items[0].kind == "rollout" and items[1].label is None and items[1].history == []
    p.write_text(json.dumps({"statement": "S"}) + "\n")
    with pytest.raises(KeyError):
        je.load_labeled_items(str(p))


def test_cli_stub_end_to_end(tmp_path):
    pf = tmp_path / "probs.jsonl"
    rows = [{"id": f"p{i}", "statement": f"Claim {i}", "label": bool(i % 2), "reference": REF + f"\n\nExtra {i}."}
            for i in range(3)] + [{"id": "noref", "statement": "z", "label": True}]
    pf.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "out"
    assert eval_judge.main(["--problems", str(pf), "--stub", "--output-dir", str(out), "--seed", "3"]) == 0
    for f in ("items.jsonl", "metrics.json", "report.txt", "comparison.txt"):
        assert (out / f).exists()
    m = json.loads((out / "metrics.json").read_text())
    assert m["build_stats"]["n_skipped_no_reference"] == 1
    assert m["n"] == len((out / "items.jsonl").read_text().splitlines())
    first = json.loads((out / "items.jsonl").read_text().splitlines()[0])
    assert {"raw_output", "judgement", "correct", "kind"} <= set(first)
    # stub accepts everything -> misses premature verdicts
    assert m["per_kind"]["premature_verdict"]["accept_rate"] == 1.0


def test_cli_labeled_only(tmp_path):
    lf = tmp_path / "l.jsonl"
    lf.write_text(json.dumps({"statement": "S", "step": "this is bad step", "gold_valid": False}) + "\n")
    out = tmp_path / "o"
    assert eval_judge.main(["--stub", "--labeled-file", str(lf), "--output-dir", str(out)]) == 0
    assert json.loads((out / "metrics.json").read_text())["recall_invalid"] == 1.0

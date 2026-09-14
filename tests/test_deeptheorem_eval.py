"""Tests for DeepTheorem prompt and verdict-format compatibility."""

from evaluation.benchmarks.deeptheorem_eval import DeepTheoremEval, _extract_verdict
from evaluation.benchmarks.deeptheorem_judge import DeepTheoremJudgeEval


def test_extract_verdict_accepts_training_boxed_format():
    assert _extract_verdict(r"proof... \boxed{proved}") == "proved"
    assert _extract_verdict(r"proof... \boxed{\text{disproved}}") == "disproved"


def test_extract_verdict_accepts_legacy_verdict_format():
    assert _extract_verdict("Verdict: PROVED") == "proved"
    assert _extract_verdict("Verdict: DISPROVED") == "disproved"


def test_eval_parser_uses_last_verdict():
    benchmark = DeepTheoremEval()
    assert benchmark.parse_output(r"The claim is discussed. \boxed{disproved}") is False
    assert benchmark.parse_output("Earlier: Verdict: DISPROVED\nVerdict: PROVED") is True


def test_judge_parser_accepts_boxed_verdict_and_steps():
    benchmark = DeepTheoremJudgeEval(verifier=object())
    parsed = benchmark.parse_output(
        r"<step>The argument establishes the claim.</step>\boxed{proved}"
    )
    assert parsed["verdict"] == "PROVED"
    assert parsed["format_valid"] is True
    assert len(parsed["steps"]) == 1


def test_eval_prompt_matches_training_output_style():
    prompt = DeepTheoremEval().get_user_prompt("Prove or disprove that P.")
    assert "<step>...</step>" in prompt
    assert r"\boxed{proved}" in prompt
    assert "Verdict: PROVED" not in prompt
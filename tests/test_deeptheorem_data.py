"""Unit tests for src/data/deeptheorem.py parsing logic (no network/datasets needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.deeptheorem import (
    DeepTheoremColumns, parse_variants, to_step_format, build_sft_examples, find_variant_column,
)

# Synthetic row mirroring the CONFIRMED DeepTheorem schema: explicit pos/neg columns
# (each a {question, response} dict) + truth_value for the original.
ROW = {
    "informal_theorem": "Let $p:E\\to X$ be a covering with $X$ connected. All fibers have the same cardinality.",
    "proof": "By local triviality each fiber is locally constant.\n\nSince $X$ is connected the function is constant. Thus the statement holds.",
    "difficulty": 8.0,
    "truth_value": True,
    "domain": "Topology",
    "pos": {"question": "Prove or disprove that all fibers have the same cardinality.",
            "response": "By local triviality the cardinality is locally constant.\n\nConnectedness gives constancy.",
            "truth_value": True},
    "neg": {"question": "Prove or disprove that there exist points with different cardinalities.",
            "response": "Local triviality forbids this.\n\nHence no such points exist.",
            "truth_value": False},
}
# Legacy list-column row (fallback path)
LEGACY_ROW = {
    "variants": [
        {"question": "claim A", "response": "neutral math"},
        {"question": "claim B", "response": "neutral math"},
    ],
}


def test_pos_neg_columns():
    vs = parse_variants(ROW, include_original=False)
    assert len(vs) == 2
    pos = next(v for v in vs if v.meta.get("variant") == "pos")
    neg = next(v for v in vs if v.meta.get("variant") == "neg")
    assert pos.label is True and "all fibers" in pos.statement
    assert neg.label is False and "different cardinalities" in neg.statement
    assert pos.reference.startswith("By local triviality")


def test_label_read_from_variant_truth_value():
    # label comes from the variant dict's own truth_value, not the pos/neg position:
    # a "pos" cell whose truth_value is False must be labelled False.
    row = {"pos": {"question": "q", "response": "r", "truth_value": False}}
    vs = parse_variants(row, include_original=False)
    assert len(vs) == 1 and vs[0].label is False


def test_include_original_uses_truth_value():
    vs = parse_variants(ROW, include_original=True)
    assert len(vs) == 3
    orig = next(v for v in vs if v.is_original)
    assert orig.label is True


def test_legacy_list_fallback():
    vs = parse_variants(LEGACY_ROW, label_strategy="positional", include_original=False, warn=False)
    assert [v.label for v in vs] == [True, False]


def test_step_format():
    out = to_step_format("first para\n\nsecond para")
    assert out == "<step>first para</step>\n<step>second para</step>"


def test_build_sft_examples_shape():
    exs = build_sft_examples([ROW], include_original=False, append_verdict=True)
    assert len(exs) == 2
    e = exs[0]
    roles = [m["role"] for m in e["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert e["messages"][1]["content"].startswith("Prove or disprove the following:")
    assert e["messages"][2]["content"].endswith("PROVED") or e["messages"][2]["content"].endswith("DISPROVED")
    assert "<step>" in e["messages"][2]["content"]


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception:
            print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)

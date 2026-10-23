import json

import pytest

from scripts import build_judge_eval_set as b
from scripts import eval_judge


def ds_row(i="t1", **kw):
    r = {
        "id": i,
        "informal_theorem_qa": "Original  statement.",
        "proof": "ORIG PROOF",
        "pos": {"question": "Pos statement.", "response": "POS PROOF", "truth_value": True},
        "neg": {"question": "Neg statement.", "response": "NEG PROOF", "truth_value": False},
    }
    r.update(kw)
    return r


@pytest.mark.parametrize("suffix,variant", [
    ("_pos", "pos"), ("_neg", "neg"), ("_ori", "original"), ("_original", "original"),
    ("_entailing", "pos"), ("_contradictory", "neg"),
])
def test_suffix_parsing(suffix, variant):
    assert b.parse_id({"id": "abc_1" + suffix}) == ("abc_1", variant)


def test_no_suffix_unresolved():
    assert b.parse_id({"id": "abc", "label": True}) == ("abc", None)


def test_fields_take_precedence():
    base, var = b.parse_id({"id": "x_pos", "base_id": "real", "variant_type": "contradictory"})
    assert (base, var) == ("real", "neg")
    assert b.parse_id({"id": "x_pos", "variant_type": "original"})[1] == "original"
    assert b.parse_id({"id": "x", "base_id": "B", "variant_type": "ori"}) == ("B", "original")


def test_reference_selection_per_variant():
    rows = [
        {"id": "t1_ori", "statement": "Original statement.", "label": True},
        {"id": "t1_pos", "statement": "Pos statement.", "label": True},
        {"id": "t1_neg", "statement": "Neg statement.", "label": False},
    ]
    out, st = b.attach_references(rows, {"t1": ds_row()})
    assert [o["reference"] for o in out] == ["ORIG PROOF", "POS PROOF", "NEG PROOF"]
    assert all(o["reference_check"] == "ok" for o in out)  # whitespace-normalized
    assert out[0]["id"] == "t1_ori" and out[0]["label"] is True
    assert st["written"] == 3 and st["statement_mismatch"] == 0


def test_stringified_dict_columns():
    r = ds_row(pos=str(ds_row()["pos"]), neg=json.dumps(ds_row()["neg"]))
    rows = [{"id": "t1_pos", "statement": "Pos statement.", "label": True},
            {"id": "t1_neg", "statement": "Neg statement.", "label": False}]
    out, _ = b.attach_references(rows, {"t1": r})
    assert [o["reference"] for o in out] == ["POS PROOF", "NEG PROOF"]


def test_statement_mismatch_kept_and_flagged():
    out, st = b.attach_references([{"id": "t1_pos", "statement": "different", "label": True}],
                                  {"t1": ds_row()})
    assert out[0]["reference_check"] == "statement_mismatch"
    assert st["statement_mismatch"] == 1 and st["written"] == 1


def test_skip_counting():
    rows = [
        {"id": "nope_pos", "statement": "s", "label": True},          # missing base id
        {"id": "t1_neg", "statement": "s", "label": False},           # missing response
        {"id": "t1", "statement": "s", "label": True},                # unresolved variant
        {"id": "t1_pos", "statement": "Pos statement.", "label": True},
    ]
    r = ds_row(neg={"question": "q", "response": ""})
    out, st = b.attach_references(rows, {"t1": r})
    assert st["skipped"] == {"missing_base_id": 1, "missing_response": 1, "unresolved_variant": 1}
    assert st["total"] == 4 and st["written"] == 1 and len(out) == 1


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_main_end_to_end_and_input_untouched(tmp_path):
    ev = tmp_path / "eval.jsonl"
    out = tmp_path / "out.jsonl"
    _write(ev, [{"id": "t1_pos", "statement": "Pos statement.", "label": True}])
    before = ev.read_text()
    rc = b.main(["--eval-file", str(ev), "--output", str(out)], row_source=lambda n, s: [ds_row()])
    assert rc == 0 and ev.read_text() == before
    assert json.loads(out.read_text())["reference"] == "POS PROOF"


def test_main_refuses_overwrite(tmp_path):
    ev = tmp_path / "eval.jsonl"
    _write(ev, [{"id": "t1_pos", "statement": "s", "label": True}])
    before = ev.read_text()
    assert b.main(["--eval-file", str(ev), "--output", str(ev)], row_source=lambda n, s: []) == 2
    assert ev.read_text() == before


def test_main_nonzero_when_nothing_written(tmp_path):
    ev = tmp_path / "eval.jsonl"
    out = tmp_path / "out.jsonl"
    _write(ev, [{"id": "zzz_pos", "statement": "s", "label": True}])
    assert b.main(["--eval-file", str(ev), "--output", str(out)], row_source=lambda n, s: []) == 1
    assert not out.exists()


def test_eval_judge_zero_reference_error(tmp_path):
    p = tmp_path / "noref.jsonl"
    _write(p, [{"id": "a_pos", "statement": "s", "label": True}])
    with pytest.raises(SystemExit) as e:
        eval_judge.main(["--stub", "--problems", str(p), "--output-dir", str(tmp_path / "o")])
    assert "build_judge_eval_set.py" in str(e.value)


def test_eval_judge_default_problems():
    assert eval_judge.parse_args([]).problems == "data/deeptheorem_judge_eval.jsonl"

"""Torch-free tests: continuation index math, OOM bookkeeping, resume state, early-stop OOM
rule, ref_mode / resume_from config wiring."""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.training.rl.grpo_tree import dropout_modules_to_eval


def test_dropout_modules_to_eval_selects_only_dropout():
    class Dropout: ...
    class Identity: ...
    class Linear: ...
    class Model:
        def __init__(self, ms): self._ms = ms
        def modules(self): return iter(self._ms)

    d1, d2, idt, lin = Dropout(), Dropout(), Identity(), Linear()
    assert dropout_modules_to_eval(Model([lin, d1, idt, d2])) == [d1, d2]
    assert dropout_modules_to_eval(Model([lin, idt]), is_dropout=lambda m: m is idt) == [idt]

from src.training.rl.early_stop import EarlyStopConfig, EarlyStopper
from src.training.rl.grpo_tree import GRPOTreeTrainer, TreeGRPOConfig, is_oom_error
from src.training.rl.train_utils import (
    TrainerState, continuation_span, cuda_indices, decode_py_random_state,
    encode_py_random_state, first_adapter_name, format_peak_mem, load_trainer_state,
    logits_to_keep_count, oom_skip_step, parse_ref_mode, problems_to_skip, publish_copy,
    resolve_ref_mode, resume_plan, save_trainer_state, trainable_param_names,
    truncate_metrics_after,
)

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "training", "configs")


# ---------------------------------------------------------------- continuation index math
def _old_positions(L, p_len):
    # old code: logits[:, :-1] -> positions 0..L-2, then tok_logp[p_len-1:]
    return list(range(L - 1))[p_len - 1:]


@pytest.mark.parametrize("L,p", [(10, 4), (5, 1), (6, 5), (100, 37), (3, 2)])
def test_continuation_span_matches_old_slice(L, p):
    start, stop = continuation_span(L, p)
    assert list(range(start, stop)) == _old_positions(L, p)
    assert stop - start == L - p  # one logit per continuation token
    # logits_to_keep path: keep last k positions, drop the final one
    k = logits_to_keep_count(L, p)
    kept = list(range(L))[L - k:][:-1]
    assert kept == _old_positions(L, p)


def test_continuation_span_nothing_to_score():
    assert continuation_span(4, 4) is None
    assert continuation_span(3, 7) is None


# ---------------------------------------------------------------- OOM bookkeeping
def test_oom_skip_step_decision():
    assert not oom_skip_step(0, 8, 0.25)
    assert not oom_skip_step(2, 8, 0.25)   # exactly 25% -> not MORE than
    assert oom_skip_step(3, 8, 0.25)
    assert not oom_skip_step(0, 0, 0.25)
    assert oom_skip_step(1, 1, 0.25)


def test_is_oom_error_by_name():
    class OutOfMemoryError(RuntimeError):
        pass

    assert is_oom_error(OutOfMemoryError("x"))
    assert not is_oom_error(RuntimeError("x"))


# ---------------------------------------------------------------- early-stop OOM rule
def _m(step, oom=0, rollout=0):
    return dict(step=step, loss=0.1, kl=0.01, grad_norm=1.0, n_samples=10,
                oom_skipped=oom, rollout_oom_skipped=rollout)


def test_early_stop_consecutive_oom_updates():
    es = EarlyStopper(EarlyStopConfig(max_oom_updates=3))
    assert es.update(_m(1, oom=1)) is None
    assert es.update(_m(2, oom=2)) is None
    d = es.update(_m(3, rollout=1))  # rollout OOMs count too
    assert d is not None and d.stop and "OOM" in d.reason and d.hard


def test_early_stop_oom_streak_resets_and_can_be_disabled():
    es = EarlyStopper(EarlyStopConfig(max_oom_updates=3))
    for i, oom in enumerate([1, 1, 0, 1, 1, 0], 1):
        assert es.update(_m(i, oom=oom)) is None
    off = EarlyStopper(EarlyStopConfig(max_oom_updates=0))
    for i in range(1, 20):
        assert off.update(_m(i, oom=5)) is None


def test_early_stop_warn_mode_does_not_stop():
    es = EarlyStopper(EarlyStopConfig(max_oom_updates=1, mode="warn"))
    d = es.update(_m(1, oom=1))
    assert d is not None and not d.stop


def test_early_stop_state_roundtrip():
    es = EarlyStopper()
    for i in range(1, 9):
        es.update(dict(step=i, loss=0.1, kl=0.01, grad_norm=1.0, n_samples=5, mean_reward=0.4,
                       mean_steps=4.0, mean_cont_tokens=100.0, premature_frac=0.0,
                       first_try_valid_rate=0.8, oom_skipped=1 if i > 6 else 0))
    state = json.loads(json.dumps(es.state_dict()))
    es2 = EarlyStopper()
    es2.load_state_dict(state)
    assert es2.state_dict() == es.state_dict()
    assert es2.oom_streak == es.oom_streak == 2


# ---------------------------------------------------------------- trainer_state / resume
def test_trainer_state_roundtrip(tmp_path):
    random.seed(123)
    random.random()
    rng = {"python": encode_py_random_state(random.getstate())}
    st = TrainerState(step=42, problem_index=170, epoch=0, n_problems=1000, batch_problems=4,
                      early_stop=EarlyStopper().state_dict(), last_healthy_step=40,
                      last_healthy_path="x/last-healthy", rng=rng)
    p = tmp_path / "trainer_state.json"
    save_trainer_state(str(p), st)
    back = load_trainer_state(str(p))
    assert back == st
    nxt = random.random()
    random.setstate(decode_py_random_state(back.rng["python"]))
    assert random.random() == nxt  # RNG restored exactly


def test_trainer_state_ignores_unknown_keys():
    s = TrainerState(step=1).to_json()
    d = json.loads(s)
    d["future_field"] = 1
    assert TrainerState.from_json(json.dumps(d)).step == 1


def test_resume_plan_and_problem_skip():
    # 10 problems, batch 4, 2 epochs; 3 batches done => 10 problems (end of epoch 0)
    assert resume_plan(10, 10, 2) == (1, 0)
    assert resume_plan(8, 10, 2) == (0, 8)
    assert resume_plan(14, 10, 2) == (1, 4)
    assert resume_plan(20, 10, 2)[0] == 2  # >= epochs -> nothing left
    assert resume_plan(0, 10, 2) == (0, 0)
    assert problems_to_skip(14, 10, 2) == 14
    assert problems_to_skip(99, 10, 2) == 20
    assert problems_to_skip(-3, 10, 2) == 0


def test_truncate_metrics_after(tmp_path):
    p = tmp_path / "train_metrics.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in
                         [{"step": 1}, {"step": 2}, {"step": 3}, {"early_stop": True, "step": 4}]))
    assert truncate_metrics_after(str(p), 2) == 2
    assert [json.loads(l)["step"] for l in p.read_text().splitlines()] == [1, 2]
    assert os.path.exists(str(p) + ".pre-resume.bak")
    assert truncate_metrics_after(str(p), 2) == 0
    assert truncate_metrics_after(str(tmp_path / "nope.jsonl"), 1) == 0


def test_publish_copy_replaces_dir(tmp_path):
    src, dst = tmp_path / "checkpoint-5", tmp_path / "checkpoint-latest"
    src.mkdir()
    (src / "a.txt").write_text("new")
    dst.mkdir()
    (dst / "stale.txt").write_text("old")
    publish_copy(str(src), str(dst))
    assert (dst / "a.txt").read_text() == "new" and not (dst / "stale.txt").exists()
    assert not (tmp_path / "checkpoint-latest.tmp").exists()


def test_trainer_resume_cursor_defaults():
    from src.training.rl.stubs import StubJudge, StubPolicy  # noqa: F401
    t = GRPOTreeTrainer(None, None, StubPolicy(), StubJudge(), cfg=TreeGRPOConfig(batch_problems=4))
    st = t._trainer_state()
    assert (st.step, st.problem_index, st.epoch, st.batch_problems) == (0, 0, 0, 4)
    TrainerState.from_json(st.to_json())  # serializable incl. RNG capture


# ---------------------------------------------------------------- ref mode / adapter helpers
def test_parse_and_resolve_ref_mode():
    assert parse_ref_mode(None) == "adapter"
    assert parse_ref_mode("Separate") == "separate"
    with pytest.raises(ValueError):
        parse_ref_mode("bogus")
    assert resolve_ref_mode("adapter", True, True) == ("adapter", None)
    mode, warn = resolve_ref_mode("adapter", False, True)
    assert mode == "separate" and warn
    mode, warn = resolve_ref_mode("adapter", True, False)
    assert mode == "separate" and warn
    assert resolve_ref_mode("separate", True, True) == ("separate", None)


def test_optimizer_params_exclude_ref_adapter():
    named = [
        ("base.q_proj.lora_A.default.weight", True),
        ("base.q_proj.lora_B.default.weight", True),
        ("base.q_proj.lora_A.ref.weight", True),   # set_adapter('ref') leaked requires_grad
        ("base.q_proj.lora_B.ref.weight", True),
        ("base.q_proj.base_layer.weight", False),
    ]
    assert trainable_param_names(named, "ref") == [
        "base.q_proj.lora_A.default.weight", "base.q_proj.lora_B.default.weight"]
    assert len(trainable_param_names(named, None)) == 4


def test_first_adapter_name():
    assert first_adapter_name("default") == "default"
    assert first_adapter_name(["default", "ref"]) == "default"
    assert first_adapter_name(None) == "default"


def test_cuda_indices_and_peak_mem_format():
    assert cuda_indices("cuda:1", "cuda", "cpu", None, "cuda:1", 2) == [0, 1, 2]
    assert cuda_indices("cpu") == []
    assert format_peak_mem({1: 2 * 2**30, 0: 3 * 2**30 // 2}) == {
        "peak_mem_gib_cuda0": 1.5, "peak_mem_gib_cuda1": 2.0}


# ---------------------------------------------------------------- config wiring
@pytest.mark.parametrize("fname", ["rl_tree.yaml", "rl_tree_from_sft.yaml"])
def test_yaml_wiring(fname):
    yaml = pytest.importorskip("yaml")
    rl = yaml.safe_load(open(os.path.join(CFG_DIR, fname)))["rl"]
    assert parse_ref_mode(rl["ref_mode"]) == "adapter"
    assert "resume_from" in rl and rl["resume_from"] is None
    # early_stop yaml mirrors dataclass defaults exactly (incl. new max_oom_updates)
    from dataclasses import asdict
    assert asdict(EarlyStopConfig.from_dict(rl["early_stop"])) == asdict(EarlyStopConfig())
    assert rl["early_stop"]["max_oom_updates"] == EarlyStopConfig().max_oom_updates == 5
    assert rl["tree"]["max_oom_skip_frac"] == TreeGRPOConfig().max_oom_skip_frac == 0.25
    TreeGRPOConfig(**rl["tree"])  # no unknown keys


def test_both_yamls_share_new_keys():
    yaml = pytest.importorskip("yaml")
    a, b = (yaml.safe_load(open(os.path.join(CFG_DIR, f)))["rl"]
            for f in ("rl_tree.yaml", "rl_tree_from_sft.yaml"))
    assert a["early_stop"] == b["early_stop"]
    assert a["ref_mode"] == b["ref_mode"] and a["resume_from"] == b["resume_from"]

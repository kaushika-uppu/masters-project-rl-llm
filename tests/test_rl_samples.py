"""Tests for sample building, problem bridging, and trainer batch assembly (no torch)."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.training.rl.types import Problem
from src.training.rl.stubs import StubPolicy, StubJudge
from src.training.rl.rollout import RolloutEngine
from src.training.rl.sample_builder import build_samples, default_prompt
from src.training.rl.grpo_tree import GRPOTreeTrainer, TreeGRPOConfig
from src.training.rl.problems import load_problems_jsonl

PROBLEM = Problem(id="t1", statement="all fibers have the same cardinality", label=True)


def test_build_samples_basic():
    eng = RolloutEngine(StubPolicy(), StubJudge(), max_depth=4, retry_budget=2)
    tree, trajs = eng.build_tree(PROBLEM, group_size=3)
    samples = build_samples(tree, trajs, PROBLEM)
    assert samples, "should produce training samples"
    # every sample has a prompt, a step, and a positive weight
    assert all(s.prompt and s.step and s.weight > 0 for s in samples)
    # the step from root onto the correct (PROVED) branch should carry higher advantage
    # than the step onto the wrong (DISPROVED) branch
    root_steps = [s for s in samples if s.meta["parent"] == tree.root_key]
    advs = {s.step: s.advantage for s in root_steps}
    assert advs.get("A", -9) > advs.get("B", 9)


def test_prompt_render_contains_history():
    p = default_prompt(PROBLEM, ["first", "second"])
    assert "first" in p and "second" in p and PROBLEM.statement in p


def test_samples_carry_chat_messages_and_continuation():
    # the trainer scores `continuation` under `messages` (same rendering the policy used)
    eng = RolloutEngine(StubPolicy(), StubJudge(), max_depth=4, retry_budget=2)
    tree, trajs = eng.build_tree(PROBLEM, group_size=3)
    samples = build_samples(tree, trajs, PROBLEM)
    s = samples[0]
    assert s.messages and [m["role"] for m in s.messages][:2] == ["system", "user"]
    # a non-verdict step is re-wrapped in <step>...</step>; a verdict step stays bare
    if s.step.upper().startswith("VERDICT"):
        assert s.continuation == s.step
    else:
        assert s.continuation == f"<step>{s.step}</step>"


def test_trainer_build_batch_no_torch():
    trainer = GRPOTreeTrainer(model=None, tokenizer=None, policy=StubPolicy(), judge=StubJudge(),
                              cfg=TreeGRPOConfig(group_size=3, max_depth=4))
    samples = trainer.build_batch([PROBLEM])
    assert len(samples) > 0


def test_trainer_build_batch_parallel():
    trainer = GRPOTreeTrainer(model=None, tokenizer=None, policy=StubPolicy(), judge=StubJudge(),
                              cfg=TreeGRPOConfig(group_size=3, max_depth=4, num_workers=2))
    probs = [PROBLEM, Problem(id="t2", statement="another claim", label=False)]
    samples = trainer.build_batch(probs)
    assert len(samples) > 0


def test_load_problems_jsonl():
    rows = [
        {"id": 1, "statement": "claim A", "label": True, "reference": "ref"},
        {"id": 2, "input": "claim B", "output": False},   # alias keys
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        path = f.name
    probs = load_problems_jsonl(path)
    os.unlink(path)
    assert len(probs) == 2
    assert probs[0].statement == "claim A" and probs[0].label is True
    assert probs[1].statement == "claim B" and probs[1].label is False


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

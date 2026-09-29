import importlib.util
import json

import numpy as np


def _load_rollout_module():
    spec = importlib.util.spec_from_file_location(
        "generate_rollouts", "scripts/rl_training_dataset/generate_rollouts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_grader_module():
    spec = importlib.util.spec_from_file_location(
        "grade_and_filter", "scripts/rl_training_dataset/grade_and_filter.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_save_rollout_results_converts_numpy_scalars(tmp_path):
    module = _load_rollout_module()
    output = tmp_path / "rollouts_chunk_0.json"
    results = [{"base_id": np.int64(42), "difficulty": np.int64(3), "rollouts": []}]

    recovery = module.save_rollout_results(results, str(output))

    assert json.loads(output.read_text())[0]["base_id"] == 42
    assert json.loads(output.read_text())[0]["difficulty"] == 3
    assert json.loads((tmp_path / "rollouts_chunk_0.json.recovery.jsonl").read_text())["base_id"] == 42
    assert recovery.endswith(".recovery.jsonl")


def test_save_rollout_results_preserves_recovery_if_final_write_fails(tmp_path, monkeypatch):
    module = _load_rollout_module()
    output = tmp_path / "rollouts_chunk_0.json"
    output.write_text("[{\"old\": true}]\n")
    results = [{"base_id": np.int64(42), "rollouts": ["saved"]}]

    def fail_replace(source, destination):
        raise OSError("simulated final publish failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)

    try:
        module.save_rollout_results(results, str(output))
    except OSError:
        pass
    else:
        raise AssertionError("expected simulated publish failure")

    assert json.loads(output.read_text()) == [{"old": True}]
    recovery = tmp_path / "rollouts_chunk_0.json.recovery.jsonl"
    assert json.loads(recovery.read_text())["rollouts"] == ["saved"]


def test_grader_uses_recovery_when_final_json_is_corrupt(tmp_path):
    module = _load_grader_module()
    (tmp_path / "rollouts_chunk_0.json").write_text("[{broken")
    (tmp_path / "rollouts_chunk_0.json.recovery.jsonl").write_text(
        '{"base_id": 42, "rollouts": []}\n'
    )

    rows = module.load_rollout_rows(str(tmp_path))

    assert rows == [{"base_id": 42, "rollouts": []}]
"""Pure (torch-free) helpers for the tree-GRPO trainer: index math, OOM bookkeeping,
KL-reference mode resolution, resume/trainer-state serialization, memory-telemetry
formatting. Everything here is unit-testable without torch/peft (tests/test_train_utils.py).
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Optional

REF_MODES = ("adapter", "separate")
REF_ADAPTER_NAME = "ref"
TRAINER_STATE_FILE = "trainer_state.json"
OPTIMIZER_FILE = "optimizer.pt"
TRAINER_STATE_VERSION = 1


# --------------------------------------------------------------------------- logits index math
def continuation_span(total_len: int, p_len: int) -> Optional[tuple[int, int]]:
    """Logit positions (start, stop_exclusive) that score the continuation tokens.

    Sequence ids[0:L]; the prompt is ids[0:p_len]; continuation tokens are ids[p_len:L].
    Token ids[t] is predicted by the logits at position t-1, so we need logit positions
    p_len-1 .. L-2 (inclusive) = range(p_len-1, L-1). Same slice the old code produced via
    `logits[:, :-1][p_len-1:]`. None when there is nothing to score."""
    p_len = max(int(p_len), 1)
    if total_len <= p_len:
        return None
    return p_len - 1, total_len - 1


def logits_to_keep_count(total_len: int, p_len: int) -> int:
    """`logits_to_keep=k` keeps the LAST k positions. We need positions p_len-1..L-2, so keep
    p_len-1..L-1 (k = L-p_len+1) and drop the final row afterwards (`[:, :-1]`)."""
    p_len = max(int(p_len), 1)
    return total_len - (p_len - 1)


def continuation_target_span(total_len: int, p_len: int) -> tuple[int, int]:
    """Target token slice (start, stop) in ids: ids[p_len:L]."""
    return max(int(p_len), 1), total_len


# --------------------------------------------------------------------------- OOM bookkeeping
def oom_skip_step(n_oom: int, n_attempted: int, max_frac: float) -> bool:
    """True when strictly more than `max_frac` of the batch's samples OOMed -> skip opt.step()."""
    if n_attempted <= 0 or n_oom <= 0:
        return False
    return (n_oom / n_attempted) > max_frac


def oom_exception_types() -> tuple:
    """torch.OutOfMemoryError (torch>=2.5) and torch.cuda.OutOfMemoryError (older)."""
    import torch

    types = []
    for obj in (torch, getattr(torch, "cuda", None)):
        t = getattr(obj, "OutOfMemoryError", None)
        if isinstance(t, type) and t not in types:
            types.append(t)
    return tuple(types) or (MemoryError,)


# --------------------------------------------------------------------------- KL reference mode
def parse_ref_mode(value) -> str:
    v = "adapter" if value is None else str(value).strip().lower()
    if v not in REF_MODES:
        raise ValueError(f"rl.ref_mode must be one of {REF_MODES}, got {value!r}")
    return v


def resolve_ref_mode(requested: str, policy_is_peft: bool, ref_path_has_adapter: bool
                     ) -> tuple[str, Optional[str]]:
    """-> (effective_mode, warning_or_None). 'adapter' needs a PeftModel policy and a PEFT
    adapter checkpoint to load as the frozen 'ref' adapter; otherwise fall back to 'separate'."""
    requested = parse_ref_mode(requested)
    if requested == "separate":
        return "separate", None
    if not policy_is_peft:
        return "separate", ("ref_mode='adapter' needs a PEFT policy (full fine-tune?) -> "
                            "falling back to ref_mode='separate' (needs rl.ref_device memory)")
    if not ref_path_has_adapter:
        return "separate", ("ref checkpoint has no adapter_config.json (not a PEFT adapter dir) "
                            "-> falling back to ref_mode='separate'")
    return "adapter", None


def is_adapter_param(name: str, adapter: str) -> bool:
    """True if a PEFT parameter name belongs to adapter `adapter` (lora_A.<adapter>.weight,
    modules_to_save.<adapter>.weight, ...)."""
    return f".{adapter}." in name or name.endswith(f".{adapter}")


def trainable_param_names(named_requires_grad: list, ref_adapter: Optional[str]) -> list[str]:
    """Names of the params the optimizer should own: requires_grad and NOT the ref adapter."""
    out = []
    for n, rg in named_requires_grad:
        if ref_adapter and is_adapter_param(n, ref_adapter):
            continue
        if rg:
            out.append(n)
    return out


def first_adapter_name(active) -> str:
    """PEFT's active_adapter is a str (new) or list[str] / str (old); return one name."""
    if isinstance(active, (list, tuple)):
        return str(active[0]) if active else "default"
    return str(active) if active else "default"


# --------------------------------------------------------------------------- telemetry
def cuda_indices(*devices) -> list[int]:
    """Sorted unique CUDA device indices from 'cuda:1' / 'cuda' / 1 / torch.device-likes.
    cpu / None are ignored."""
    out = set()
    for d in devices:
        if d is None:
            continue
        if isinstance(d, bool):
            continue
        if isinstance(d, int):
            out.add(d)
            continue
        s = str(d)
        if s == "cuda":
            out.add(0)
        elif s.startswith("cuda:"):
            try:
                out.add(int(s.split(":", 1)[1]))
            except ValueError:
                pass
    return sorted(out)


def format_peak_mem(bytes_by_device: dict) -> dict:
    """{idx: bytes} -> flat jsonl-friendly {'peak_mem_gib_cuda<idx>': GiB}."""
    return {f"peak_mem_gib_cuda{i}": round(b / 2**30, 3) for i, b in sorted(bytes_by_device.items())}


# --------------------------------------------------------------------------- resume / state
def encode_py_random_state(state) -> list:
    """random.getstate() -> JSON-safe nested lists."""
    version, internal, gauss = state
    return [version, list(internal), gauss]


def decode_py_random_state(obj) -> tuple:
    version, internal, gauss = obj
    return (version, tuple(internal), gauss)


@dataclass
class TrainerState:
    step: int = 0                 # completed optimizer updates
    problem_index: int = 0        # problems consumed, cumulative across epochs
    epoch: int = 0
    n_problems: int = 0
    batch_problems: int = 0
    early_stop: dict = field(default_factory=dict)
    last_healthy_step: Optional[int] = None
    last_healthy_path: Optional[str] = None
    rng: dict = field(default_factory=dict)
    version: int = TRAINER_STATE_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, s: str) -> "TrainerState":
        d = json.loads(s)
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


def save_trainer_state(path: str, state: TrainerState) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(state.to_json())
    os.replace(tmp, path)


def load_trainer_state(path: str) -> TrainerState:
    with open(path) as f:
        return TrainerState.from_json(f.read())


def resume_plan(problem_index: int, n_problems: int, epochs: int) -> tuple[int, int]:
    """(start_epoch, offset_within_epoch) from the cumulative problems-consumed counter.
    start_epoch >= epochs means everything was already processed."""
    if n_problems <= 0:
        return epochs, 0
    return problem_index // n_problems, problem_index % n_problems


def problems_to_skip(problem_index: int, n_problems: int, epochs: int) -> int:
    """How many problems are skipped (not re-run) on resume, capped at the run total."""
    return max(0, min(problem_index, n_problems * epochs))


def truncate_metrics_after(path: str, step: int) -> int:
    """Drop train_metrics.jsonl records with record['step'] > step (written after the
    checkpoint we resume from). A backup is kept next to the file. Returns #dropped."""
    if not os.path.exists(path):
        return 0
    kept, dropped = [], 0
    with open(path) as f:
        lines = f.readlines()
    for ln in lines:
        try:
            rec = json.loads(ln)
        except ValueError:
            kept.append(ln)
            continue
        s = rec.get("step") if isinstance(rec, dict) else None
        if isinstance(s, (int, float)) and s > step:
            dropped += 1
        else:
            kept.append(ln)
    if dropped:
        shutil.copyfile(path, path + ".pre-resume.bak")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.writelines(kept)
        os.replace(tmp, path)
    return dropped


def publish_copy(src: str, dst: str) -> None:
    """Replace directory `dst` with a copy of `src` (checkpoint-latest / last-healthy)
    without re-serializing the weights. Copy to a temp dir first so a crash mid-copy
    never leaves `dst` half-written."""
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.replace(tmp, dst)

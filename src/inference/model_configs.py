"""Config table for inference functions that load a merged training checkpoint.

Each entry maps an inference-function name to the arguments
`load_training_inference()` needs (checkpoint path, system prompt, cot flag).
Adding a new checkpoint/prompt/cot combination means adding an entry here,
not a new file.
"""

from src.inference.constants import DEEPTHEOREM_SYSTEM_PROMPT, RIDDLEBENCH_SYSTEM_PROMPT

_HELPFUL_ASSISTANT_PROMPT = "You are a helpful assistant."

TRAINING_INFERENCE_CONFIGS = {
    "rl_from_base_merged": {
        "model_path": "./checkpoints/rl_from_base_merged",
        "system_prompt": _HELPFUL_ASSISTANT_PROMPT,
        "use_cot_3shot": False,
    },
    "rl_from_base_merged_cot": {
        "model_path": "./checkpoints/rl_from_base_merged",
        "system_prompt": RIDDLEBENCH_SYSTEM_PROMPT,
        "use_cot_3shot": True,
    },
    "rl_from_base_merged_deeptheorem": {
        "model_path": "./checkpoints/rl_from_base_merged",
        "system_prompt": DEEPTHEOREM_SYSTEM_PROMPT,
        "use_cot_3shot": False,
    },
    "rl_sft_merged": {
        "model_path": "./checkpoints/rl_sft_merged",
        "system_prompt": _HELPFUL_ASSISTANT_PROMPT,
        "use_cot_3shot": False,
    },
    "rl_sft_merged_cot": {
        "model_path": "./checkpoints/rl_sft_merged",
        "system_prompt": RIDDLEBENCH_SYSTEM_PROMPT,
        "use_cot_3shot": True,
    },
    "rl_sft_merged_deeptheorem": {
        "model_path": "./checkpoints/rl_sft_merged",
        "system_prompt": DEEPTHEOREM_SYSTEM_PROMPT,
        "use_cot_3shot": False,
    },
    "rl_tree_base_checkpoint_350_merged": {
        "model_path": "./checkpoints/rl_tree_base_complete",
        "system_prompt": RIDDLEBENCH_SYSTEM_PROMPT,
        "use_cot_3shot": False,
    },
    "rl_tree_base_checkpoint_350_merged_cot": {
        "model_path": "./checkpoints/rl_tree_base_checkpoint_350_merged",
        "system_prompt": RIDDLEBENCH_SYSTEM_PROMPT,
        "use_cot_3shot": True,
    },
    "rl_tree_base_checkpoint_350_merged_deeptheorem": {
        "model_path": "./checkpoints/rl_tree_base_checkpoint_350_merged",
        "system_prompt": DEEPTHEOREM_SYSTEM_PROMPT,
        "use_cot_3shot": False,
    },
    "rl_tree_sft_merged": {
        "model_path": "./checkpoints/rl_tree_sft_merged",
        "system_prompt": _HELPFUL_ASSISTANT_PROMPT,
        "use_cot_3shot": False,
    },
    "rl_tree_sft_merged_cot": {
        "model_path": "./checkpoints/rl_tree_sft_merged",
        "system_prompt": RIDDLEBENCH_SYSTEM_PROMPT,
        "use_cot_3shot": True,
    },
    "rl_tree_sft_merged_deeptheorem": {
        "model_path": "./checkpoints/rl_tree_sft_merged",
        "system_prompt": DEEPTHEOREM_SYSTEM_PROMPT,
        "use_cot_3shot": False,
    },
}

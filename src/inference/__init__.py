"""Inference and prompting logic."""

from src.inference.test_inference import dummy_inference
from src.inference.base_inference import BaseInference
from src.inference.cot_3shot import cot_3shot
from src.inference.limo_inference import limo
from src.inference.deepseek_r1_inference import deepseek_r1
from src.inference.sft_inference import sft_merged
from src.inference.qwen_base_inference import qwen_base
from src.inference.rl_sft_merged import rl_sft_merged
from src.inference.rl_sft_merged_cot import rl_sft_merged_cot
from src.inference.rl_from_base_merged import rl_from_base_merged
from src.inference.rl_from_base_merged_cot import rl_from_base_merged_cot
from src.inference.rl_tree_from_base_checkpoint_350_merged import rl_tree_base_checkpoint_350_merged
from src.inference.rl_tree_from_base_checkpoint_350_merged_cot import rl_tree_base_checkpoint_350_merged_cot
from src.inference.rl_tree_from_sft import rl_tree_sft_merged
from src.inference.rl_tree_from_sft_cot import rl_tree_sft_merged_cot
from src.inference.rl_sft_merged_deeptheorem import rl_sft_merged_deeptheorem
from src.inference.rl_from_base_merged_deeptheorem import rl_from_base_merged_deeptheorem
from src.inference.rl_tree_from_base_checkpoint_350_merged_deeptheorem import rl_tree_base_checkpoint_350_merged_deeptheorem
from src.inference.rl_tree_from_sft_deeptheorem import rl_tree_sft_merged_deeptheorem


__all__ = [
    "dummy_inference", 
    "BaseInference", 
    "cot_3shot", 
    "limo", 
    "deepseek_r1", 
    "sft_merged", 
    "qwen_base", 
    "rl_sft_merged", 
    "rl_from_base_merged", 
    "rl_from_base_merged_cot",
    "rl_tree_base_checkpoint_350_merged",
    "rl_tree_base_checkpoint_350_merged_cot",
    "rl_tree_sft_merged",
    "rl_tree_sft_merged_cot",
    "rl_sft_merged_deeptheorem",
    "rl_from_base_merged_deeptheorem",
    "rl_tree_base_checkpoint_350_merged_deeptheorem",
    "rl_tree_sft_merged_deeptheorem",

    ]

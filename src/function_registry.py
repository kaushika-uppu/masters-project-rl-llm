# src/function_registry.py
# keeps track of all functions that can be used for inference

from typing import Dict, Callable

# Lazy imports - only import when functions are actually requested
def _get_registry() -> Dict[str, Callable[[str], str]]:
    """Lazy load inference functions to avoid importing heavy dependencies."""
    from src.inference import (
        dummy_inference,
        cot_3shot,
        limo,
        deepseek_r1,
        sft_merged,
        qwen_base,
        rl_sft_merged,
        rl_sft_merged_cot,
        rl_from_base_merged,
        rl_from_base_merged_cot,
        rl_tree_base_checkpoint_350_merged,
        rl_tree_base_checkpoint_350_merged_cot,
        rl_tree_sft_merged,
        rl_tree_sft_merged_cot,
        rl_sft_merged_deeptheorem,
        rl_from_base_merged_deeptheorem,
        rl_tree_base_checkpoint_350_merged_deeptheorem,
        rl_tree_sft_merged_deeptheorem,

    )
    return {
        "dummy": dummy_inference,
        "cot_3shot": cot_3shot,
        "limo": limo,
        "deepseek_r1": deepseek_r1,
        "sft_merged": sft_merged,
        "qwen_base": qwen_base,
        "rl_sft_merged": rl_sft_merged,
        "rl_sft_merged": rl_sft_merged_cot,
        "rl_from_base_merged": rl_from_base_merged,
        "rl_from_base_merged_cot": rl_from_base_merged_cot,
        "rl_tree_base_checkpoint_350_merged": rl_tree_base_checkpoint_350_merged,
        "rl_tree_base_checkpoint_350_merged_cot": rl_tree_base_checkpoint_350_merged_cot,
        "rl_tree_sft_merged": rl_tree_sft_merged,
        "rl_tree_sft_merged_cot": rl_tree_sft_merged_cot,
        "rl_sft_merged_deeptheorem": rl_sft_merged_deeptheorem,
        "rl_from_base_merged_deeptheorem": rl_from_base_merged_deeptheorem,
        "rl_tree_base_checkpoint_350_merged_deeptheorem": rl_tree_base_checkpoint_350_merged_deeptheorem,
        "rl_tree_sft_merged_deeptheorem": rl_tree_sft_merged_deeptheorem,


    }

def get_available_functions() -> list[str]:
    """Return list of available inference function names."""
    return ["dummy", 
        "cot_3shot", 
        "limo", 
        "deepseek_r1", 
        "sft_merged", 
        "qwen_base", 
        "rl_sft_merged", 
        "rl_sft_merged_cot", 
        "rl_from_base_merged", 
        "rl_from_base_merged_cot", 
        "rl_tree_base_checkpoint_350_merged",
        "rl_tree_base_checkpoint_350_merged_cot",
        "rl_tree_sft_merged",
        "rl_tree_sft_merged_cot",
        "rl_sft_merged_deeptheorem",
        "rl_from_base_merged_deeptheorem",
        "rl_tree_base_checkpoint_350_merged_deeptheorem",
        "rl_tree_sft_merged_deeptheorem"

        ]

def get_inference_function(name: str) -> Callable[[str], str]:
    """Get an inference function by name. Uses lazy imports to avoid loading heavy dependencies."""
    available = get_available_functions()
    if name not in available:
        raise ValueError(f"Unknown inference function '{name}'. Available: {available}")

    registry = _get_registry()

    if name == 'limo':
        from src.inference.limo_inference import limo
        return limo()
    if name == 'deepseek_r1':
        from src.inference.deepseek_r1_inference import deepseek_r1
        return deepseek_r1()

    if name == 'sft_merged':
        from src.inference.sft_inference import sft_merged
        return sft_merged()

    if name == 'qwen_base':
        from src.inference.qwen_base_inference import qwen_base
        return qwen_base()

    if name == 'rl_sft_merged':
        from src.inference import rl_sft_merged
        return rl_sft_merged()
    
    if name == 'rl_sft_merged_cot':
        from src.inference import rl_sft_merged_cot
        return rl_sft_merged_cot()
    
    if name == 'rl_from_base_merged':
        from src.inference import rl_from_base_merged
        return rl_from_base_merged()

    if name == 'rl_from_base_merged_cot':
        from src.inference import rl_from_base_merged_cot
        return rl_from_base_merged_cot()

    if name == 'rl_tree_base_checkpoint_350_merged':
        from src.inference import rl_tree_base_checkpoint_350_merged
        return rl_tree_base_checkpoint_350_merged()
    
    if name == 'rl_tree_base_checkpoint_350_merged_cot':
        from src.inference import rl_tree_base_checkpoint_350_merged_cot 
        return rl_tree_base_checkpoint_350_merged_cot()   

    if name == 'rl_tree_sft_merged':
        from src.inference import rl_tree_sft_merged
        return rl_tree_sft_merged()
    
    if name == 'rl_tree_sft_merged_cot':
        from src.inference import rl_tree_sft_merged_cot
        return rl_tree_sft_merged_cot()   

    if name == 'rl_sft_merged_deeptheorem':
        from src.inference import rl_sft_merged_deeptheorem
        return rl_sft_merged_deeptheorem()

    if name == 'rl_from_base_merged_deeptheorem':
        from src.inference import rl_from_base_merged_deeptheorem
        return rl_from_base_merged_deeptheorem()

    if name == 'rl_tree_base_checkpoint_350_merged_deeptheorem':
        from src.inference import rl_tree_base_checkpoint_350_merged_deeptheorem
        return rl_tree_base_checkpoint_350_merged_deeptheorem()    

    if name == 'rl_tree_sft_merged_deeptheorem':
        from src.inference import rl_tree_sft_merged_deeptheorem
        return rl_tree_sft_merged_deeptheorem()
    
    return registry[name]

# Training Configuration Guide

This directory contains YAML configuration files for training models with SFT (Supervised Fine-Tuning) and RL (Reinforcement Learning).

## Quick Start

```bash
# Run training with a config file
python -m src.training.run src/training/configs/your_config.yaml
```

## Configuration Structure

### Required Sections

#### `model` (Required)
Specify **exactly ONE** of these options:

```yaml
model:
  # Option 1: Load from HuggingFace
  name: "Qwen/Qwen2.5-7B-Instruct"
  
  # Option 2: Load from checkpoint
  # from_checkpoint: "checkpoints/sft_deeptheorem"
  # is_peft_checkpoint: true  # Set true if it has LoRA adapters
  
  # Option 3: Load from saved config
  # from_config: "configs/qwen2.5_7b_instruct.yaml"
  
  # Model parameters
  torch_dtype: "bfloat16"  # Options: "bfloat16", "float16", "float32", "auto"
  device_map: "auto"       # Options: "auto", "cuda", "cpu"
  
  # Optional additional kwargs
  additional_kwargs:
    trust_remote_code: false
```

#### Training Mode (At least ONE required)

Include `sft` section, `rl` section, or both:

```yaml
sft:
  dataset: "deeptheorem"  # Options: "deeptheorem", "gsm8k"
  output_dir: "checkpoints/sft_deeptheorem"

  # Optional: Limit training examples (useful for testing)
  max_samples: 100  # Use only first 100 examples. Omit for full dataset.

  # Optional hyperparameters (defaults shown)
  num_train_epochs: 2
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  warmup_steps: 0
  max_grad_norm: 1.0
  optim: "adamw_torch_fused"
  weight_decay: 0.0
  bf16: true
  fp16: false
  logging_steps: 50
  save_steps: 500
  save_total_limit: 3
  gradient_checkpointing: true
  max_length: 1024
  packing: false

rl:
  dataset: "deeptheorem"  # Options: "deeptheorem", "gsm8k"
  output_dir: "checkpoints/rl_deeptheorem"
  algorithm: "dpo"  # Not yet implemented
```

### Optional Sections

#### `lora` (Optional - Recommended for 7B+ models)

```yaml
lora:
  r: 16                    # Rank: 4, 8, 16, 32, 64, 128
  alpha: 32                # Typically 2x rank
  dropout: 0.05            # 0.0, 0.05, 0.1
  bias: "none"             # "none", "all", "lora_only"
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
```

**Note:** LoRA is NOT applied if loading from a PEFT checkpoint (`is_peft_checkpoint: true`).

## Configuration Examples

### Example 1: SFT with LoRA
```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  torch_dtype: "bfloat16"
  device_map: "auto"

lora:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
  dropout: 0.05

sft:
  dataset: "deeptheorem"
  output_dir: "checkpoints/sft_deeptheorem"
```

### Example 2: SFT then RL Pipeline
```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  torch_dtype: "bfloat16"
  device_map: "auto"

lora:
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

sft:
  dataset: "deeptheorem"
  output_dir: "checkpoints/sft_deeptheorem"

rl:
  dataset: "deeptheorem"
  output_dir: "checkpoints/rl_deeptheorem"
  algorithm: "dpo"
```

**Behavior:** SFT runs first, then RL automatically loads the SFT checkpoint and continues training.

### Example 3: Continue from Checkpoint
```yaml
model:
  from_checkpoint: "checkpoints/sft_deeptheorem/checkpoint-1000"
  is_peft_checkpoint: true  # Important if it has LoRA
  torch_dtype: "bfloat16"
  device_map: "auto"

sft:
  dataset: "deeptheorem"
  output_dir: "checkpoints/sft_deeptheorem_continued"
```

### Example 4: Full Model Training (No LoRA)
```yaml
model:
  name: "Qwen/Qwen2.5-0.5B-Instruct"  # Use smaller model for full training
  torch_dtype: "bfloat16"
  device_map: "auto"

# No lora section = full model training

sft:
  dataset: "gsm8k"
  output_dir: "checkpoints/sft_gsm8k_full"
```

## Available Datasets

| Dataset | Source | Size | Description |
|---------|--------|------|-------------|
| `deeptheorem` | Jiahao004/DeepTheorem | 121K | Math theorem-proof pairs, filtered to difficulty ≤ 7.0, formatted with step tags |
| `gsm8k` | openai/gsm8k | 7.5K | Grade school math word problems |

## Memory Requirements

| Configuration | VRAM Required | Recommended GPU |
|---------------|---------------|-----------------|
| 7B + LoRA (r=16) | ~20-24 GB | RTX 4090 (24GB), A100 (40GB) |
| 7B Full Fine-tune | ~60-80 GB | A100 (80GB), Multi-GPU |
| 0.5B Full Fine-tune | ~4-6 GB | RTX 3060 (12GB) |

## Best Practices

1. **Use LoRA for large models** (7B+) to save memory
2. **Use bfloat16** if your GPU supports it (most modern GPUs)
3. **Start with small epochs** to test your configuration
4. **Monitor GPU memory** with `nvidia-smi` during training
5. **Save checkpoints frequently** (configured in `sft.py`)

## SFT Hyperparameters Reference

All hyperparameters are optional and have sensible defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_samples` | None | Limit number of training examples (useful for testing). If None, uses full dataset. |
| `num_train_epochs` | 2 | Number of training epochs |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 4 | Steps to accumulate gradients (effective batch = batch_size × accum_steps) |
| `learning_rate` | 2e-5 | Learning rate (use ~1e-4 for LoRA, ~2e-5 for full fine-tuning) |
| `warmup_steps` | 0 | Number of warmup steps for learning rate scheduler |
| `max_grad_norm` | 1.0 | Maximum gradient norm for clipping |
| `optim` | "adamw_torch_fused" | Optimizer (options: "adamw_torch", "adamw_torch_fused", "adamw_8bit") |
| `weight_decay` | 0.0 | Weight decay for regularization |
| `bf16` | true | Use bfloat16 mixed precision (recommended for modern GPUs) |
| `fp16` | false | Use float16 mixed precision (use if bf16 not supported) |
| `logging_steps` | 50 | Log metrics every N steps |
| `logging_strategy` | "steps" | When to log (options: "steps", "epoch") |
| `save_steps` | 500 | Save checkpoint every N steps |
| `save_strategy` | "steps" | When to save (options: "steps", "epoch") |
| `save_total_limit` | 3 | Maximum number of checkpoints to keep |
| `eval_strategy` | "no" | Evaluation strategy (options: "no", "steps", "epoch") |
| `gradient_checkpointing` | true | Enable gradient checkpointing (saves memory, slower) |
| `max_length` | 1024 | Maximum sequence length |
| `packing` | false | Pack multiple examples into one sequence |
| `report_to` | "none" | Logging service (options: "none", "tensorboard", "wandb") |

## Current Limitations

- RL training is not yet implemented (TODO)

## Troubleshooting

**Windows UTF-8 Encoding Error (TRL library):**
```
UnicodeDecodeError: 'charmap' codec can't decode byte 0x81
```
Solution - Use the batch file launcher:
```bash
.\train.bat src/training/configs/test_small.yaml
```
Or set environment variable before running:
```bash
$env:PYTHONUTF8=1; python -m src.training.run src/training/configs/test_small.yaml
```

**Out of Memory:**
- Enable LoRA if not already using it
- Reduce batch size (per_device_train_batch_size)
- Increase gradient accumulation steps
- Use a smaller model
- Set `max_samples: 100` for testing

**Padding token errors:**
- Automatically handled - tokenizer.pad_token set to tokenizer.eos_token

**PEFT loading errors:**
- Make sure `is_peft_checkpoint: true` when loading LoRA checkpoints
- Don't include `lora` section when loading PEFT checkpoints

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedModel
from trl import SFTConfig, SFTTrainer

from src.data.deeptheorem import build_sft_examples, split_by_difficulty

from .constants import DEEPTHEOREM_SYSTEM_PROMPT
from .datasets import get_dataset


def run_sft(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    dataset: str,
    output_dir: str,
    sft_config: dict = None,
) -> None:
    """Run SFT training with configurable hyperparameters.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        dataset: Dataset name to use
        output_dir: Where to save the trained model
        sft_config: Optional dict of SFT hyperparameters. Defaults are used if not specified.
            Special parameters:
            - max_samples: Limit number of training examples (for testing)
    """
    # ds = get_dataset(dataset)
    # filt_ds = format_dataset(ds, dataset)

    # getting dataset from file path
    if dataset.endswith(".jsonl"):
        print(f"Loading local dataset from {dataset}")
        ds = load_dataset("json", data_files=dataset)["train"]
    else:
        ds = get_dataset(dataset)

    filt_ds = format_dataset(ds, dataset)

    # Get config with defaults
    config = sft_config or {}
    ds = get_dataset(dataset)
    filt_ds = format_dataset(ds, dataset, config)

    # Limit dataset size if max_samples is specified (useful for testing)
    max_samples = config.get("max_samples")
    if max_samples is not None:
        print(f"Limiting training to {max_samples} examples (for testing purposes)")
        filt_ds = filt_ds.select(range(min(max_samples, len(filt_ds))))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=filt_ds,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=config.get("num_train_epochs", 2),
            per_device_train_batch_size=config.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            learning_rate=config.get("learning_rate", 2e-5),
            warmup_steps=config.get("warmup_steps", 0),
            max_grad_norm=config.get("max_grad_norm", 1.0),
            optim=config.get("optim", "adamw_torch_fused"),
            weight_decay=config.get("weight_decay", 0.0),
            bf16=config.get("bf16", True),
            fp16=config.get("fp16", False),
            logging_steps=config.get("logging_steps", 50),
            logging_strategy=config.get("logging_strategy", "steps"),
            save_steps=config.get("save_steps", 500),
            save_strategy=config.get("save_strategy", "steps"),
            save_total_limit=config.get("save_total_limit", 3),
            eval_strategy=config.get("eval_strategy", "no"),
            gradient_checkpointing=config.get("gradient_checkpointing", True),
            max_length=config.get("max_length", 2048),
            packing=config.get("packing", False),
            report_to=config.get("report_to", "none"),
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)


def format_dataset(ds: Dataset, dataset: str) -> Dataset:
    """Get subset of dataset and format for use in SFT. Default is unchanged."""
    if dataset.endswith(".jsonl"):
        return format_local_dataset(ds)
    if dataset == "deeptheorem":
        return get_deeptheorem(ds)

    if dataset == "gsm8k":
        return ds

    return ds  # default: no filtering


def format_local_dataset(ds: Dataset) -> Dataset:
    """Format local JSONL reasoning rollouts for use in SFT."""

    def format_example(row):
        raw_proof = str(row.get("proof", ""))
        # split proof into paragraphs and wrap each in <step> tags
        paragraphs = raw_proof.split("\n\n")
        proof_steps = "\n".join([f"<step>{p}</step>" for p in paragraphs])

        return {
            "messages": [
                {
                    "role": "system",
                    "content": DEEPTHEOREM_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": f"Prove the following:\n{row['informal_theorem']}",
                },
                {"role": "assistant", "content": proof_steps},
            ]
        }

    return ds.map(format_example)


def get_deeptheorem(ds: Dataset, config: dict = None) -> Dataset:
    """Format DeepTheorem for SFT.

    task_format (config):
      "prove_or_disprove" (default) - matches the RL task; expands each theorem into its
          true/false variants (verdict + <step> proof). Use this for the SFT baseline /
          RL init so there is no train->RL distribution shift.
      "prove_only" (legacy) - original behaviour: prove the always-true statement.
    """
    config = config or {}
    threshold = config.get("difficulty_threshold", 7.0)
    task_format = config.get("task_format", "prove_or_disprove")

    easy, _hard = split_by_difficulty(ds, threshold=threshold)

    if task_format == "prove_or_disprove":
        examples = build_sft_examples(
            easy,
            include_original=config.get("include_original", False),
            label_strategy=config.get("label_strategy", "auto"),
            append_verdict=config.get("append_verdict", True),
        )
        if not examples:
            raise ValueError(
                f"DeepTheorem SFT produced 0 examples (difficulty <= {threshold}, "
                f"task_format='prove_or_disprove'). Check difficulty_threshold and that "
                f"the dataset has the expected pos/neg variant columns."
            )
        return Dataset.from_list(examples)

    # legacy prove-only path
    def format_example(row):
        paragraphs = row["proof"].split("\n\n")
        proof_steps = "\n".join([f"<step>{p}</step>" for p in paragraphs])
        return {
            "messages": [
                {"role": "system", "content": DEEPTHEOREM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Prove the following:\n{row['informal_theorem']}",
                },
                {"role": "assistant", "content": proof_steps},
            ]
        }

    return easy.map(format_example)

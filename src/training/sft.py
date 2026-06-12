from .datasets import get_dataset, DatasetName
from transformers import PreTrainedModel, AutoTokenizer
from datasets import Dataset
from trl import SFTTrainer, SFTConfig


def run_sft(model: PreTrainedModel, tokenizer: AutoTokenizer, dataset: DatasetName, output_dir: str) -> None:
    ds = get_dataset(dataset)
    filt_ds = format_dataset(ds, dataset)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=filt_ds,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            bf16=True,
            logging_steps=50,
        ),
    )
    trainer.train()
    trainer.save_model(output_dir)


def format_dataset(ds: Dataset, dataset: DatasetName) -> Dataset:
    """Get subset of dataset and format for use in SFT. Default is unchanged."""
    if dataset == "deeptheorem":
        return get_deeptheorem(ds)

    if dataset == "gsm8k":
        return ds

    return ds  # default: no filtering


DEEPTHEOREM_SYSTEM_PROMPT = (
    "You are a mathematical reasoning assistant. Work through proofs step by step."
)


def get_deeptheorem(ds: Dataset) -> Dataset:
    # filter to easy examples
    ds = ds.filter(lambda x: x["difficulty"] <= 7.0)
    
    # put examples into training format
    def format_example(row):
        proof_steps = row["proof"].split("\n\n").map(lambda x: f"<step>{x}</step>").join("\n")
        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You axre a mathematical reasoning assistant. Work through proofs step by step.",
                },
                {
                    "role": "user",
                    "content": f"Prove the following:\n{row['informal_theorem']}",
                },
                {"role": "assistant", "content": proof_steps},
            ]
        }

    return ds.map(format_example)

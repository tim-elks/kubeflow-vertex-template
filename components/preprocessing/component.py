"""Preprocessing pipeline component.

This component tokenizes a raw Hugging Face dataset produced by data_ingestion
and writes train/test splits ready for the Trainer.
"""

import os

import click
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer


def preprocess(
    input_path: str,
    output_train_path: str,
    output_test_path: str,
    model_name: str = "distilbert-base-uncased",
    text_column: str = "text",
    test_size: float = 0.2,
    max_length: int = 128,
) -> None:
    """Tokenize the dataset and split into train/test sets.

    Args:
        input_path:        Path to the Arrow dataset directory from data_ingestion.
        output_train_path: Destination directory for the tokenized training split.
        output_test_path:  Destination directory for the tokenized test split.
        model_name:        Hugging Face model name used to load the tokenizer.
        text_column:       Name of the text column in the dataset.
        test_size:         Fraction of data to reserve for testing (0–1).
        max_length:        Maximum token sequence length.
    """
    print(f"[preprocessing] Loading dataset from: {input_path}")
    dataset = load_from_disk(input_path)

    # Flatten to a single Dataset if the loaded object is already a DatasetDict
    if isinstance(dataset, DatasetDict):
        # Merge all splits, then re-split deterministically
        dataset = dataset["train"] if "train" in dataset else next(iter(dataset.values()))

    print(f"[preprocessing] Tokenizing with: {model_name!r} (max_length={max_length})")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # -----------------------------------------------------------------
    # Adjust the tokenization logic for your task, e.g. add label mapping,
    # handle sentence-pairs, etc.
    # -----------------------------------------------------------------
    def tokenize(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=[text_column])

    split = dataset.train_test_split(test_size=test_size, seed=42)

    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)
    split["train"].save_to_disk(output_train_path)
    split["test"].save_to_disk(output_test_path)

    print(f"[preprocessing] Train split ({len(split['train'])} rows) → {output_train_path}")
    print(f"[preprocessing] Test split  ({len(split['test'])} rows) → {output_test_path}")


@click.command()
@click.option("--input-path", required=True, help="Arrow dataset directory from data_ingestion")
@click.option("--output-train-path", required=True, help="Directory for tokenized training split")
@click.option("--output-test-path", required=True, help="Directory for tokenized test split")
@click.option(
    "--model-name",
    default="distilbert-base-uncased",
    show_default=True,
    help="Hugging Face model name for the tokenizer",
)
@click.option(
    "--text-column", default="text", show_default=True, help="Dataset column containing text"
)
@click.option("--test-size", type=float, default=0.2, show_default=True, help="Test fraction (0–1)")
@click.option(
    "--max-length", type=int, default=128, show_default=True, help="Max token sequence length"
)
def main(
    input_path: str,
    output_train_path: str,
    output_test_path: str,
    model_name: str,
    text_column: str,
    test_size: float,
    max_length: int,
) -> None:
    preprocess(
        input_path,
        output_train_path,
        output_test_path,
        model_name,
        text_column,
        test_size,
        max_length,
    )


if __name__ == "__main__":
    main()
